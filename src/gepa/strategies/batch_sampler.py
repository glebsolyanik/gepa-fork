# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random
from collections import Counter
from typing import Callable, Hashable, Protocol
from collections import defaultdict

from gepa.core.adapter import DataInst
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState


class BatchSampler(Protocol[DataId, DataInst]):
    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]: ...


class EpochShuffledBatchSampler(BatchSampler[DataId, DataInst]):
    """
    Mirrors the original batching logic:
    - Shuffle ids each epoch
    - Pad to minibatch size with least frequent ids
    - Deterministic via state.rng1
    """

    def __init__(self, minibatch_size: int, rng: random.Random | None = None):
        self.minibatch_size = minibatch_size
        self.shuffled_ids: list[DataId] = []
        self.epoch = -1
        self.id_freqs = Counter()
        self.last_trainset_size = 0
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def _update_shuffled(self, loader: DataLoader[DataId, DataInst]):
        all_ids = list(loader.all_ids())
        trainset_size = len(loader)
        self.last_trainset_size = trainset_size

        if trainset_size == 0:
            self.shuffled_ids = []
            self.id_freqs = Counter()
            return

        self.shuffled_ids = list(all_ids)
        self.rng.shuffle(self.shuffled_ids)
        self.id_freqs = Counter(self.shuffled_ids)

        mod = trainset_size % self.minibatch_size
        num_to_pad = (self.minibatch_size - mod) if mod != 0 else 0
        if num_to_pad > 0:
            for _ in range(num_to_pad):
                selected_id = self.id_freqs.most_common()[::-1][0][0]
                self.shuffled_ids.append(selected_id)
                self.id_freqs[selected_id] += 1

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]:
        trainset_size = len(loader)
        if trainset_size == 0:
            raise ValueError("Cannot sample a minibatch from an empty loader.")

        base_idx = state.i * self.minibatch_size
        curr_epoch = 0 if self.epoch == -1 else base_idx // max(len(self.shuffled_ids), 1)

        needs_refresh = not self.shuffled_ids or trainset_size != self.last_trainset_size or curr_epoch > self.epoch
        if needs_refresh:
            self.epoch = curr_epoch
            self._update_shuffled(loader)

        assert len(self.shuffled_ids) >= self.minibatch_size
        assert len(self.shuffled_ids) % self.minibatch_size == 0

        base_idx = base_idx % len(self.shuffled_ids)
        end_idx = base_idx + self.minibatch_size
        assert end_idx <= len(self.shuffled_ids)
        return self.shuffled_ids[base_idx:end_idx]

class StratifiedEpochShuffledBatchSampler(BatchSampler[DataId, DataInst]):
    """
    Стратифицированный батч-семплер, который сохраняет пропорции страт в каждом батче.
    
    - Разделяет данные на страты на основе функции стратификации
    - В каждом батче сохраняет пропорции страт из исходного датасета
    - Перемешивает данные внутри каждой страты каждый эпоху
    - Детерминирован через state.rng1
    
    Args:
        minibatch_size: Размер мини-батча
        stratum_extractor: Функция, которая извлекает страту из DataInst (например, 
                          lambda inst: inst.get("metadata", {}).get("category"))
        rng: Генератор случайных чисел для воспроизводимости
    """

    def __init__(
        self,
        minibatch_size: int,
        stratum_extractor: Callable[[DataInst], Hashable],
        rng: random.Random | None = None,
    ):
        self.minibatch_size = minibatch_size
        self.stratum_extractor = stratum_extractor
        self.shuffled_ids: list[DataId] = []
        self.epoch = -1
        self.id_freqs = Counter()
        self.last_trainset_size = 0
        self.last_trainset_hash: int | None = None
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def _compute_trainset_hash(self, loader: DataLoader[DataId, DataInst]) -> int:
        """Вычисляет хеш датасета для определения изменений."""
        all_ids = list(loader.all_ids())
        return hash(tuple(sorted(all_ids)))

    def _update_shuffled(self, loader: DataLoader[DataId, DataInst]):
        """Обновляет перемешанные ids с учетом стратификации."""
        all_ids = list(loader.all_ids())
        trainset_size = len(loader)
        self.last_trainset_size = trainset_size

        if trainset_size == 0:
            self.shuffled_ids = []
            self.id_freqs = Counter()
            return

        # Загружаем данные для стратификации
        data_instances = loader.fetch(all_ids)
        id_to_instance = dict(zip(all_ids, data_instances, strict=True))

        # Разделяем на страты
        strata: dict[Hashable, list[DataId]] = defaultdict(list)
        for data_id in all_ids:
            instance = id_to_instance[data_id]
            stratum = self.stratum_extractor(instance)
            strata[stratum].append(data_id)

        # Перемешиваем каждую страту отдельно
        for stratum_ids in strata.values():
            self.rng.shuffle(stratum_ids)

        # Вычисляем пропорции страт
        stratum_proportions = {
            stratum: len(ids) / trainset_size for stratum, ids in strata.items()
        }

        # Формируем батчи с сохранением пропорций
        # Используем round-robin подход для равномерного распределения
        shuffled_ids: list[DataId] = []
        stratum_iterators = {
            stratum: iter(ids) for stratum, ids in strata.items()
        }
        stratum_counts = {stratum: len(ids) for stratum, ids in strata.items()}

        # Создаем батчи, сохраняя пропорции
        num_batches = (trainset_size + self.minibatch_size - 1) // self.minibatch_size
        for batch_idx in range(num_batches):
            batch_ids: list[DataId] = []
            
            # Для каждого батча распределяем элементы пропорционально стратам
            for stratum, proportion in stratum_proportions.items():
                num_from_stratum = max(1, int(self.minibatch_size * proportion))
                # Убеждаемся, что не берем больше, чем есть в страте
                num_from_stratum = min(num_from_stratum, stratum_counts[stratum])
                
                for _ in range(num_from_stratum):
                    try:
                        batch_ids.append(next(stratum_iterators[stratum]))
                    except StopIteration:
                        # Если страта закончилась, пропускаем
                        pass

            # Если батч не заполнен, добавляем из оставшихся страт
            while len(batch_ids) < self.minibatch_size:
                added = False
                for stratum, iterator in stratum_iterators.items():
                    try:
                        batch_ids.append(next(iterator))
                        added = True
                        break
                    except StopIteration:
                        continue
                if not added:
                    break

            # Перемешиваем батч для случайного порядка внутри батча
            self.rng.shuffle(batch_ids)
            shuffled_ids.extend(batch_ids)

        # Если последний батч неполный, дополняем его
        mod = trainset_size % self.minibatch_size
        if mod != 0:
            num_to_pad = self.minibatch_size - mod
            # Дополняем из наименее частых элементов
            id_freqs = Counter(shuffled_ids)
            for _ in range(num_to_pad):
                selected_id = id_freqs.most_common()[::-1][0][0]
                shuffled_ids.append(selected_id)
                id_freqs[selected_id] += 1

        self.shuffled_ids = shuffled_ids
        self.id_freqs = Counter(self.shuffled_ids)

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]:
        trainset_size = len(loader)
        if trainset_size == 0:
            raise ValueError("Cannot sample a minibatch from an empty loader.")

        base_idx = state.i * self.minibatch_size
        curr_epoch = 0 if self.epoch == -1 else base_idx // max(len(self.shuffled_ids), 1)

        # Проверяем, нужно ли обновить стратификацию
        trainset_hash = self._compute_trainset_hash(loader)
        needs_refresh = (
            not self.shuffled_ids
            or trainset_size != self.last_trainset_size
            or trainset_hash != self.last_trainset_hash
            or curr_epoch > self.epoch
        )

        if needs_refresh:
            self.epoch = curr_epoch
            self.last_trainset_hash = trainset_hash
            self._update_shuffled(loader)

        assert len(self.shuffled_ids) >= self.minibatch_size
        assert len(self.shuffled_ids) % self.minibatch_size == 0

        base_idx = base_idx % len(self.shuffled_ids)
        end_idx = base_idx + self.minibatch_size
        assert end_idx <= len(self.shuffled_ids)
        return self.shuffled_ids[base_idx:end_idx]