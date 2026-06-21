#!/usr/bin/env python3
"""Simulate the balanced-odd-player Old Maid rule."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path


JOKER = -1
RULE_ID = "balanced_odd_T_2kN_plus_1"
DEFAULT_PLAYERS = "2-13"
DEFAULT_TRIALS = 100000
DEFAULT_MAX_PAIRS = 26
DEFAULT_K = "auto"
DEFAULT_SEED = 20260620
DEFAULT_OUT_PREFIX = "results/balanced_odd"


PLAYER_HEADERS = [
    "rule_id(規則代號)",
    "N(玩家人數)",
    "k(每位玩家基準對數)",
    "pairs(使用對數)",
    "total_cards(總牌數)",
    "trials(模擬局數)",
    "seat(玩家座位)",
    "is_initial_odd_player(是否為開局奇數玩家)",
    "initial_hand_size(開局手牌數)",
    "initial_joker_probability(開局持有落單牌機率)",
    "loss_count(輸的次數)",
    "loss_rate(輸的比例)",
    "loss_rate_se(輸率標準誤)",
    "loss_rate_ci95_low(輸率95信賴區間下限)",
    "loss_rate_ci95_high(輸率95信賴區間上限)",
    "first_out_credit_rate(首位出局比例)",
    "initial_clear_rate(開局整理後出局比例)",
    "active_exit_rate(自己行動後出局比例)",
    "avg_rank(平均名次)",
]

CASE_HEADERS = [
    "N(玩家人數)",
    "k(每位玩家基準對數)",
    "pairs(使用對數)",
    "total_cards(總牌數)",
    "trials(模擬局數)",
    "odd_loss_rate(奇數玩家輸率)",
    "others_loss_rate_mean(其他玩家平均輸率)",
    "loss_rate_diff_odd_minus_others(奇數玩家輸率減其他玩家平均輸率)",
    "odd_first_out_rate(奇數玩家首位出局比例)",
    "others_first_out_rate_mean(其他玩家平均首位出局比例)",
    "odd_avg_rank(奇數玩家平均名次)",
    "others_avg_rank_mean(其他玩家平均名次)",
    "avg_turns(平均回合數)",
    "avg_initial_clears(平均開局整理後出局人數)",
    "passive_elimination_count(被動出局次數)",
    "invariant_violation_count(奇偶不變量違反次數)",
    "final_joker_error_count(結局落單牌檢查錯誤次數)",
]


@dataclass
class GameResult:
    loser: int
    turns: int
    ranks: list[float]
    first_out_credit: list[float]
    initial_clear: list[bool]
    active_exit: list[bool]
    passive_eliminations: int
    invariant_violations: int
    final_joker_error: bool


@dataclass
class CaseStats:
    player_rows: list[dict[str, object]]
    case_row: dict[str, object]


class ProgressBar:
    def __init__(self, total: int, label: str, width: int = 30) -> None:
        self.total = max(total, 1)
        self.label = label
        self.width = width
        self.last_percent = -1

    def update(self, completed: int) -> None:
        completed = min(max(completed, 0), self.total)
        percent = int(completed * 100 / self.total)
        if percent == self.last_percent and completed != self.total:
            return

        self.last_percent = percent
        filled = int(self.width * completed / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        sys.stdout.write(
            f"\r{self.label} [{bar}] {percent:3d}% ({completed}/{self.total})"
        )
        sys.stdout.flush()
        if completed == self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


def parse_players(value: str) -> list[int]:
    players: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start > end:
                raise ValueError(f"invalid player range: {part}")
            players.update(range(start, end + 1))
        else:
            players.add(int(part))

    if not players:
        raise ValueError("players cannot be empty")
    return sorted(players)


def parse_k(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    k = int(value)
    if k < 1:
        raise ValueError("--k must be auto or a positive integer")
    return k


def validate_case(n_players: int, trials: int, max_pairs: int, k: int) -> None:
    if n_players < 2:
        raise ValueError("N must be at least 2")
    if trials <= 0:
        raise ValueError("--trials must be positive")
    if max_pairs <= 0:
        raise ValueError("--max-pairs must be positive")
    if k < 1:
        raise ValueError("k must be at least 1")
    if k * n_players > max_pairs:
        raise ValueError(
            f"P = kN = {k * n_players} exceeds max pairs {max_pairs} "
            f"for N = {n_players}"
        )


def build_deck(pair_count: int) -> list[int]:
    deck: list[int] = []
    for pair_id in range(pair_count):
        deck.append(pair_id)
        deck.append(pair_id)
    deck.append(JOKER)
    return deck


def remove_pairs(hand: list[int]) -> list[int]:
    unmatched: list[int] = []
    for card in hand:
        if card != JOKER and card in unmatched:
            unmatched.remove(card)
        else:
            unmatched.append(card)
    return unmatched


def alive_players(hands: list[list[int]]) -> list[int]:
    return [seat for seat, hand in enumerate(hands) if hand]


def next_player_with_cards(hands: list[list[int]], start: int) -> int | None:
    n_players = len(hands)
    seat = (start + 1) % n_players
    while seat != start:
        if hands[seat]:
            return seat
        seat = (seat + 1) % n_players
    return None


def count_odd_alive_players(hands: list[list[int]]) -> int:
    return sum(1 for hand in hands if hand and len(hand) % 2 == 1)


def ranks_from_out_groups(n_players: int, out_groups: list[list[int]], loser: int) -> list[float]:
    ranks = [0.0] * n_players
    next_rank = 1

    for group in out_groups:
        group_size = len(group)
        rank = (next_rank + next_rank + group_size - 1) / 2
        for seat in group:
            ranks[seat] = rank
        next_rank += group_size

    ranks[loser] = float(n_players)
    return ranks


def first_out_credit(n_players: int, out_groups: list[list[int]]) -> list[float]:
    credit = [0.0] * n_players
    if not out_groups:
        return credit

    first_group = out_groups[0]
    share = 1.0 / len(first_group)
    for seat in first_group:
        credit[seat] = share
    return credit


def play_one_game(n_players: int, k: int, rng: random.Random) -> GameResult:
    pair_count = k * n_players
    deck = build_deck(pair_count)
    rng.shuffle(deck)

    hand_sizes = [2 * k + 1] + [2 * k] * (n_players - 1)
    hands: list[list[int]] = []
    offset = 0
    for hand_size in hand_sizes:
        hand = deck[offset : offset + hand_size]
        hands.append(remove_pairs(hand))
        offset += hand_size

    initial_clear = [not hand for hand in hands]
    active_exit = [False] * n_players
    out_groups: list[list[int]] = []
    initial_clear_group = [seat for seat, cleared in enumerate(initial_clear) if cleared]
    if initial_clear_group:
        out_groups.append(initial_clear_group)

    turns = 0
    passive_eliminations = 0
    invariant_violations = 0
    active = 0

    if count_odd_alive_players(hands) != 1:
        invariant_violations += 1

    while len(alive_players(hands)) > 1:
        if not hands[active]:
            odd_players = [
                seat for seat, hand in enumerate(hands) if hand and len(hand) % 2 == 1
            ]
            if not odd_players:
                invariant_violations += 1
                break
            active = odd_players[0]

        if count_odd_alive_players(hands) != 1:
            invariant_violations += 1

        target = next_player_with_cards(hands, active)
        if target is None:
            break

        turns += 1
        drawn_index = rng.randrange(len(hands[target]))
        drawn_card = hands[target].pop(drawn_index)
        if not hands[target]:
            passive_eliminations += 1

        if drawn_card != JOKER and drawn_card in hands[active]:
            hands[active].remove(drawn_card)
        else:
            hands[active].append(drawn_card)

        if not hands[active] and active not in initial_clear_group:
            active_exit[active] = True
            out_groups.append([active])

        if count_odd_alive_players(hands) != 1 and len(alive_players(hands)) > 1:
            invariant_violations += 1

        active = target

    remaining = alive_players(hands)
    loser = remaining[0] if remaining else 0
    final_joker_error = not (remaining and hands[loser] == [JOKER])
    ranks = ranks_from_out_groups(n_players, out_groups, loser)
    credit = first_out_credit(n_players, out_groups)

    return GameResult(
        loser=loser,
        turns=turns,
        ranks=ranks,
        first_out_credit=credit,
        initial_clear=initial_clear,
        active_exit=active_exit,
        passive_eliminations=passive_eliminations,
        invariant_violations=invariant_violations,
        final_joker_error=final_joker_error,
    )


def fmt_float(value: float) -> str:
    return f"{value:.10f}"


def binomial_se(rate: float, trials: int) -> float:
    return math.sqrt(rate * (1.0 - rate) / trials)


def simulate_case(
    n_players: int,
    k: int,
    trials: int,
    rng: random.Random,
    progress_label: str | None = None,
) -> CaseStats:
    pair_count = k * n_players
    total_cards = 2 * pair_count + 1
    initial_hand_sizes = [2 * k + 1] + [2 * k] * (n_players - 1)

    loss_counts = [0] * n_players
    first_out_credits = [0.0] * n_players
    initial_clear_counts = [0] * n_players
    active_exit_counts = [0] * n_players
    rank_sums = [0.0] * n_players
    total_turns = 0
    total_initial_clears = 0
    passive_elimination_count = 0
    invariant_violation_count = 0
    final_joker_error_count = 0
    progress = ProgressBar(trials, progress_label) if progress_label else None
    if progress:
        progress.update(0)

    for trial_index in range(1, trials + 1):
        result = play_one_game(n_players, k, rng)
        loss_counts[result.loser] += 1
        total_turns += result.turns
        passive_elimination_count += result.passive_eliminations
        invariant_violation_count += result.invariant_violations
        final_joker_error_count += int(result.final_joker_error)

        for seat in range(n_players):
            first_out_credits[seat] += result.first_out_credit[seat]
            initial_clear_counts[seat] += int(result.initial_clear[seat])
            active_exit_counts[seat] += int(result.active_exit[seat])
            rank_sums[seat] += result.ranks[seat]
            total_initial_clears += int(result.initial_clear[seat])
        if progress:
            progress.update(trial_index)

    player_rows: list[dict[str, object]] = []
    loss_rates: list[float] = []
    first_out_rates: list[float] = []
    avg_ranks: list[float] = []

    for seat in range(n_players):
        loss_rate = loss_counts[seat] / trials
        se = binomial_se(loss_rate, trials)
        ci_low = max(0.0, loss_rate - 1.96 * se)
        ci_high = min(1.0, loss_rate + 1.96 * se)
        first_out_rate = first_out_credits[seat] / trials
        avg_rank = rank_sums[seat] / trials

        loss_rates.append(loss_rate)
        first_out_rates.append(first_out_rate)
        avg_ranks.append(avg_rank)

        player_rows.append(
            {
                "rule_id(規則代號)": RULE_ID,
                "N(玩家人數)": n_players,
                "k(每位玩家基準對數)": k,
                "pairs(使用對數)": pair_count,
                "total_cards(總牌數)": total_cards,
                "trials(模擬局數)": trials,
                "seat(玩家座位)": seat,
                "is_initial_odd_player(是否為開局奇數玩家)": "是" if seat == 0 else "否",
                "initial_hand_size(開局手牌數)": initial_hand_sizes[seat],
                "initial_joker_probability(開局持有落單牌機率)": fmt_float(
                    initial_hand_sizes[seat] / total_cards
                ),
                "loss_count(輸的次數)": loss_counts[seat],
                "loss_rate(輸的比例)": fmt_float(loss_rate),
                "loss_rate_se(輸率標準誤)": fmt_float(se),
                "loss_rate_ci95_low(輸率95信賴區間下限)": fmt_float(ci_low),
                "loss_rate_ci95_high(輸率95信賴區間上限)": fmt_float(ci_high),
                "first_out_credit_rate(首位出局比例)": fmt_float(first_out_rate),
                "initial_clear_rate(開局整理後出局比例)": fmt_float(
                    initial_clear_counts[seat] / trials
                ),
                "active_exit_rate(自己行動後出局比例)": fmt_float(
                    active_exit_counts[seat] / trials
                ),
                "avg_rank(平均名次)": fmt_float(avg_rank),
            }
        )

    odd_loss_rate = loss_rates[0]
    others_loss_rate_mean = sum(loss_rates[1:]) / (n_players - 1)
    odd_first_out_rate = first_out_rates[0]
    others_first_out_rate_mean = sum(first_out_rates[1:]) / (n_players - 1)
    odd_avg_rank = avg_ranks[0]
    others_avg_rank_mean = sum(avg_ranks[1:]) / (n_players - 1)

    case_row = {
        "N(玩家人數)": n_players,
        "k(每位玩家基準對數)": k,
        "pairs(使用對數)": pair_count,
        "total_cards(總牌數)": total_cards,
        "trials(模擬局數)": trials,
        "odd_loss_rate(奇數玩家輸率)": fmt_float(odd_loss_rate),
        "others_loss_rate_mean(其他玩家平均輸率)": fmt_float(others_loss_rate_mean),
        "loss_rate_diff_odd_minus_others(奇數玩家輸率減其他玩家平均輸率)": fmt_float(
            odd_loss_rate - others_loss_rate_mean
        ),
        "odd_first_out_rate(奇數玩家首位出局比例)": fmt_float(odd_first_out_rate),
        "others_first_out_rate_mean(其他玩家平均首位出局比例)": fmt_float(
            others_first_out_rate_mean
        ),
        "odd_avg_rank(奇數玩家平均名次)": fmt_float(odd_avg_rank),
        "others_avg_rank_mean(其他玩家平均名次)": fmt_float(others_avg_rank_mean),
        "avg_turns(平均回合數)": fmt_float(total_turns / trials),
        "avg_initial_clears(平均開局整理後出局人數)": fmt_float(
            total_initial_clears / trials
        ),
        "passive_elimination_count(被動出局次數)": passive_elimination_count,
        "invariant_violation_count(奇偶不變量違反次數)": invariant_violation_count,
        "final_joker_error_count(結局落單牌檢查錯誤次數)": final_joker_error_count,
    }

    return CaseStats(player_rows=player_rows, case_row=case_row)


def write_csv(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    players: list[int],
    trials: int,
    max_pairs: int,
    seed: int,
    k_text: str,
    case_rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 抽鬼牌模擬摘要",
        "",
        "## 模擬參數",
        "",
        f"- 規則代號：`{RULE_ID}`",
        f"- 玩家數：`{','.join(str(player) for player in players)}`",
        f"- 每組模擬局數：`{trials}`",
        f"- 最大可用對數：`{max_pairs}`",
        f"- k 設定：`{k_text}`",
        f"- 隨機種子：`{seed}`",
        "",
        "## 各玩家數設定",
        "",
        "| 玩家數 N | k | 使用對數 P | 總牌數 T | 發牌型態 |",
        "|---:|---:|---:|---:|---|",
    ]

    for row in case_rows:
        n_players = int(row["N(玩家人數)"])
        k = int(row["k(每位玩家基準對數)"])
        hand_pattern = f"{2 * k + 1}, " + ", ".join(["%d" % (2 * k)] * (n_players - 1))
        lines.append(
            "| {n} | {k} | {pairs} | {total} | {pattern} |".format(
                n=n_players,
                k=k,
                pairs=row["pairs(使用對數)"],
                total=row["total_cards(總牌數)"],
                pattern=hand_pattern,
            )
        )

    lines.extend(
        [
            "",
            "## 奇數玩家與其他玩家比較",
            "",
            "| 玩家數 N | 奇數玩家輸率 | 其他玩家平均輸率 | 差值 | 奇數玩家首位出局比例 | 其他玩家平均首位出局比例 | 奇數玩家平均名次 | 其他玩家平均名次 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in case_rows:
        lines.append(
            "| {n} | {odd_loss} | {other_loss} | {diff} | {odd_first} | {other_first} | {odd_rank} | {other_rank} |".format(
                n=row["N(玩家人數)"],
                odd_loss=row["odd_loss_rate(奇數玩家輸率)"],
                other_loss=row["others_loss_rate_mean(其他玩家平均輸率)"],
                diff=row[
                    "loss_rate_diff_odd_minus_others(奇數玩家輸率減其他玩家平均輸率)"
                ],
                odd_first=row["odd_first_out_rate(奇數玩家首位出局比例)"],
                other_first=row["others_first_out_rate_mean(其他玩家平均首位出局比例)"],
                odd_rank=row["odd_avg_rank(奇數玩家平均名次)"],
                other_rank=row["others_avg_rank_mean(其他玩家平均名次)"],
            )
        )

    lines.extend(
        [
            "",
            "## 流程檢查",
            "",
            "| 玩家數 N | 平均回合數 | 平均開局整理後出局人數 | 被動出局次數 | 奇偶不變量違反次數 | 結局落單牌檢查錯誤次數 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in case_rows:
        lines.append(
            "| {n} | {turns} | {initial} | {passive} | {invariant} | {joker} |".format(
                n=row["N(玩家人數)"],
                turns=row["avg_turns(平均回合數)"],
                initial=row["avg_initial_clears(平均開局整理後出局人數)"],
                passive=row["passive_elimination_count(被動出局次數)"],
                invariant=row["invariant_violation_count(奇偶不變量違反次數)"],
                joker=row["final_joker_error_count(結局落單牌檢查錯誤次數)"],
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prompt_until_valid(
    label: str,
    default: object,
    description: str,
    convert,
):
    while True:
        raw = input(f"{label} [{default}] - {description}\n> ").strip()
        if raw == "":
            raw = str(default)
        try:
            return convert(raw)
        except ValueError as exc:
            print(f"輸入錯誤：{exc}")


def parse_positive_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} 必須是正整數")
    return parsed


def parse_nonempty_text(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} 不可空白")
    return value.strip()


def parse_players_text(value: str) -> str:
    parsed = parse_nonempty_text(value, "玩家人數")
    parse_players(parsed)
    return parsed


def parse_k_text(value: str) -> str:
    parsed = parse_nonempty_text(value, "k 設定")
    parse_k(parsed)
    return parsed


def collect_interactive_args() -> argparse.Namespace:
    print("抽鬼牌模擬程式互動模式")
    print("直接按 Enter 會使用預設值。")
    print()

    players_text = prompt_until_valid(
        "玩家人數",
        DEFAULT_PLAYERS,
        '可輸入範圍或清單，例如 "2-13"、"3,5,7"、"2-6,10"',
        parse_players_text,
    )

    trials = prompt_until_valid(
        "每個玩家數要模擬幾局",
        DEFAULT_TRIALS,
        "建議正式研究至少 100000；測試可先用 1000",
        lambda value: parse_positive_int(value, "模擬局數"),
    )
    max_pairs = prompt_until_valid(
        "最大可用對數",
        DEFAULT_MAX_PAIRS,
        "標準 52 張牌為 26 對",
        lambda value: parse_positive_int(value, "最大可用對數"),
    )
    k_text = prompt_until_valid(
        "k 設定",
        DEFAULT_K,
        '輸入 auto 代表每個 N 使用 floor(最大可用對數 / N)，或輸入正整數',
        parse_k_text,
    )

    seed = prompt_until_valid(
        "隨機種子",
        DEFAULT_SEED,
        "同一個種子會產生可重現結果",
        int,
    )
    out_prefix = prompt_until_valid(
        "輸出檔案前綴",
        DEFAULT_OUT_PREFIX,
        "會產生 *_player_stats.csv、*_case_stats.csv、*_summary.md",
        lambda value: parse_nonempty_text(value, "輸出檔案前綴"),
    )

    return argparse.Namespace(
        players=players_text,
        trials=trials,
        max_pairs=max_pairs,
        k=k_text,
        seed=seed,
        out_prefix=out_prefix,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate balanced-odd-player Old Maid games."
    )
    parser.add_argument(
        "--players", default=DEFAULT_PLAYERS, help='player counts, e.g. "2-13"'
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="trials per N")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=DEFAULT_MAX_PAIRS,
        help="maximum available pair count, 26 for a standard deck",
    )
    parser.add_argument(
        "--k",
        default=DEFAULT_K,
        help='auto or a positive integer; auto uses floor(max_pairs / N)',
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument(
        "--out-prefix",
        default=DEFAULT_OUT_PREFIX,
        help="output file prefix",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    interactive = len(sys.argv) == 1

    while True:
        if interactive:
            args = collect_interactive_args()
        else:
            args = parser.parse_args()

        try:
            players = parse_players(args.players)
            fixed_k = parse_k(args.k)

            for n_players in players:
                k = args.max_pairs // n_players if fixed_k is None else fixed_k
                validate_case(n_players, args.trials, args.max_pairs, k)
            break
        except ValueError as exc:
            if not interactive:
                parser.error(str(exc))
            print()
            print(f"參數組合錯誤：{exc}")
            print("請重新輸入一次。")
            print()

    rng = random.Random(args.seed)
    player_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []

    for n_players in players:
        k = args.max_pairs // n_players if fixed_k is None else fixed_k
        total_cards = 2 * k * n_players + 1
        progress_label = f"N={n_players} k={k} T={total_cards}"
        stats = simulate_case(n_players, k, args.trials, rng, progress_label)
        player_rows.extend(stats.player_rows)
        case_rows.append(stats.case_row)
        print(
            "N={n} k={k} T={total} trials={trials} done".format(
                n=n_players,
                k=k,
                total=stats.case_row["total_cards(總牌數)"],
                trials=args.trials,
            )
        )

    out_prefix = Path(args.out_prefix)
    player_path = out_prefix.with_name(out_prefix.name + "_player_stats.csv")
    case_path = out_prefix.with_name(out_prefix.name + "_case_stats.csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.md")

    write_csv(player_path, PLAYER_HEADERS, player_rows)
    write_csv(case_path, CASE_HEADERS, case_rows)
    write_summary(
        summary_path,
        players,
        args.trials,
        args.max_pairs,
        args.seed,
        args.k,
        case_rows,
    )

    print(f"wrote {player_path}")
    print(f"wrote {case_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
