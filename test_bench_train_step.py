import unittest

from experiments.bench_train_step import (
    benchmark_accounting,
    kda_layer_counts,
    optimizer_due,
    summarize_samples,
)


class BenchTrainStepAccountingTest(unittest.TestCase):
    def test_optimizer_runs_once_per_accumulation_window(self):
        self.assertFalse(optimizer_due(0, 2))
        self.assertTrue(optimizer_due(1, 2))
        self.assertFalse(optimizer_due(2, 2))
        self.assertTrue(optimizer_due(3, 2))

    def test_real_window_throughput_counts_all_microbatch_tokens(self):
        stats = benchmark_accounting(
            batch_size=12,
            seq_len=1024,
            accumulation_steps=2,
            elapsed_seconds=2.0,
        )
        self.assertEqual(stats["tokens"], 24_576)
        self.assertEqual(stats["tokens_per_second"], 12_288)
        self.assertEqual(stats["seconds_per_microbatch"], 1.0)

    def test_1080m_has_six_main_kda_and_gqa_mtp(self):
        # Qwen MTP 是 full-attn，不再占用 KDA 层。
        self.assertEqual(kda_layer_counts(num_hidden_layers=8, mtp_depth=1), (6, 0))

    def test_periodic_optimizer_summary_keeps_refresh_cost(self):
        median, minimum, mean = summarize_samples([1.0] * 7 + [9.0])
        self.assertEqual(median, 1.0)
        self.assertEqual(minimum, 1.0)
        self.assertEqual(mean, 2.0)


if __name__ == "__main__":
    unittest.main()
