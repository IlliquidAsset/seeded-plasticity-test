"""Environment hygiene: build the REAL RSS watchdog in the run interpreter.

Every run_pool test injects a fake ``_Watch()``, so the production
``RSSWatch`` (which imports psutil) was never exercised in the review
interpreter. This test closes that gap (t_631a163b root cause, t_bf15bb55).

It deliberately does NOT skip when psutil is missing: an interpreter that
cannot construct the real watchdog cannot run the diagnostic, so the suite
must fail with ModuleNotFoundError there.

No simulation, no seed, no scientific file is touched.
"""

import os

import run_stage2
from run_stage2 import RSS_LIMIT, RSSWatch


def test_real_rss_watch_constructs_and_samples_once():
    watch = RSSWatch()  # real class: imports psutil, binds psutil.Process()

    assert type(watch) is run_stage2.RSSWatch
    assert watch.limit == RSS_LIMIT == 8 * 1024 ** 3
    assert watch.proc.pid == os.getpid()

    total = watch.sample()

    assert isinstance(total, int) and total > 0
    assert watch.peak_total == total
    assert 0 < watch.peak_single <= total
    assert watch.max_children >= 0
    assert watch.exceeded is False  # a pytest process is far below 8 GiB
    assert not watch.t.is_alive()  # sampling once never starts the abort thread
