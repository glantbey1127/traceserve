from collections import deque
from dataclasses import dataclass, field

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


@dataclass(slots=True)
class Preemption:
    seq_id: int
    cached_before: int
    free_before: int
    free_after: int
    reason: str = "decode_block_shortage"


@dataclass(slots=True)
class ScheduleEvents:
    preemptions: list[Preemption] = field(default_factory=list)
    cache_hits: dict[int, int] = field(default_factory=dict)


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_resident_seqs = config.max_resident_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.decode_reserve_blocks = config.decode_reserve_blocks
        self.max_consecutive_prefill_steps = config.max_consecutive_prefill_steps
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, self.block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.consecutive_prefill_steps = 0
        self.last_events = ScheduleEvents()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    @property
    def resident_count(self):
        return len(self.running) + sum(bool(seq.block_table) for seq in self.waiting)

    def schedule(self) -> tuple[list[Sequence], bool]:
        self.last_events = ScheduleEvents()
        if self.is_finished():
            return [], False
        # A self-preemption can release enough capacity for a new prefill attempt.
        for _ in range(2):
            scheduled: list[Sequence] = []
            used_tokens = 0
            allow_prefill = (
                not self.running
                or self.consecutive_prefill_steps < self.max_consecutive_prefill_steps
            )
            if allow_prefill:
                for seq in tuple(self.waiting):
                    if len(scheduled) == self.max_num_seqs:
                        break
                    remaining = self.max_num_batched_tokens - used_tokens
                    if not remaining:
                        break
                    if not seq.block_table:
                        residents = self.resident_count
                        if residents >= self.max_resident_seqs:
                            break
                        reserve = self.decode_reserve_blocks if residents else 0
                        cached = self.block_manager.can_allocate(seq, reserve_blocks=reserve)
                        if cached == -1:
                            break
                        tokens = seq.num_tokens - cached * self.block_size
                    else:
                        tokens = seq.num_tokens - seq.num_cached_tokens
                        cached = 0
                    if scheduled and remaining < tokens:
                        break
                    if not seq.block_table:
                        self.block_manager.allocate(seq, cached)
                        self.last_events.cache_hits[seq.seq_id] = cached * self.block_size
                    seq.num_scheduled_tokens = min(tokens, remaining)
                    used_tokens += seq.num_scheduled_tokens
                    if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                        seq.status = SequenceStatus.RUNNING
                        self.waiting.remove(seq)
                        self.running.append(seq)
                    scheduled.append(seq)
                if scheduled:
                    self.consecutive_prefill_steps += 1
                    return scheduled, True

            self.consecutive_prefill_steps = 0
            while self.running and len(scheduled) < min(self.max_num_seqs, self.max_num_batched_tokens):
                seq = self.running.popleft()
                while not self.block_manager.can_append(seq):
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    seq.num_scheduled_tokens = 1
                    seq.is_prefill = False
                    self.block_manager.may_append(seq)
                    scheduled.append(seq)
            if scheduled:
                self.running.extendleft(reversed(scheduled))
                return scheduled, False
        raise RuntimeError("scheduler cannot make progress; check KV capacity and admission configuration")

    def preempt(self, seq: Sequence):
        cached_before = seq.num_cached_tokens
        free_before = len(self.block_manager.free_block_ids)
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.last_events.preemptions.append(Preemption(
            seq.seq_id, cached_before, free_before, len(self.block_manager.free_block_ids),
        ))
        self.waiting.appendleft(seq)

    def cancel(self, seq: Sequence, *, failed: bool = False):
        if seq.status == SequenceStatus.WAITING:
            self.waiting.remove(seq)
        else:
            self.running.remove(seq)
        if seq.block_table:
            self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.FAILED if failed else SequenceStatus.CANCELLED
        seq.finish_reason = "error" if failed else "cancelled"

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            end = seq.num_cached_tokens + seq.num_scheduled_tokens
            seq.computed_high_watermark = max(seq.computed_high_watermark, end)
            seq.num_cached_tokens = end
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            eos = not seq.ignore_eos and token_id == self.eos
            if eos or seq.num_completion_tokens == seq.max_tokens:
                seq.finish_reason = "stop" if eos else "length"
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
