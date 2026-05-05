#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POMDP 信念状態更新デモ
単一ホスト侵入シナリオにおける信念状態の更新過程と方策比較を示す概念実装。

このスクリプトはネットワーク接続・実攻撃を一切行いません。
研究で設計した POMDP モデルの数理的コアのみを示しています。

使い方:
    python demo_belief.py

依存:
    pip install numpy
"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Tuple

# ---------------------------------------------------------------------------
# モデル定義
# ---------------------------------------------------------------------------

STATES = ["initial", "probing", "compromised", "detected"]
ACTIONS = ["wait", "ssh_scan", "vuln_scan", "ssh_login", "exploit"]
OBSERVATIONS = ["none", "service_found", "vuln_found", "auth_success", "alert"]

S = {s: i for i, s in enumerate(STATES)}
A = {a: i for i, a in enumerate(ACTIONS)}
O = {o: i for i, o in enumerate(OBSERVATIONS)}

ns, na, no = len(STATES), len(ACTIONS), len(OBSERVATIONS)

# ---------------------------------------------------------------------------
# 遷移確率 T[a, s, s'] = P(s' | s, a)
# ---------------------------------------------------------------------------

T = np.zeros((na, ns, ns))

# wait: 状態変化なし
T[A["wait"]] = np.eye(ns)

# ssh_scan
T[A["ssh_scan"], S["initial"],    S["initial"]]    = 0.3
T[A["ssh_scan"], S["initial"],    S["probing"]]    = 0.5
T[A["ssh_scan"], S["initial"],    S["detected"]]   = 0.2
T[A["ssh_scan"], S["probing"],    S["probing"]]    = 1.0
T[A["ssh_scan"], S["compromised"],S["compromised"]]= 1.0
T[A["ssh_scan"], S["detected"],   S["detected"]]   = 1.0

# vuln_scan
T[A["vuln_scan"], S["initial"],    S["initial"]]    = 0.6
T[A["vuln_scan"], S["initial"],    S["probing"]]    = 0.3
T[A["vuln_scan"], S["initial"],    S["detected"]]   = 0.1
T[A["vuln_scan"], S["probing"],    S["probing"]]    = 0.85
T[A["vuln_scan"], S["probing"],    S["detected"]]   = 0.15
T[A["vuln_scan"], S["compromised"],S["compromised"]]= 1.0
T[A["vuln_scan"], S["detected"],   S["detected"]]   = 1.0

# ssh_login
T[A["ssh_login"], S["initial"],    S["initial"]]    = 0.7
T[A["ssh_login"], S["initial"],    S["detected"]]   = 0.3
T[A["ssh_login"], S["probing"],    S["probing"]]    = 0.3
T[A["ssh_login"], S["probing"],    S["compromised"]]= 0.5
T[A["ssh_login"], S["probing"],    S["detected"]]   = 0.2
T[A["ssh_login"], S["compromised"],S["compromised"]]= 1.0
T[A["ssh_login"], S["detected"],   S["detected"]]   = 1.0

# exploit
T[A["exploit"], S["initial"],    S["initial"]]    = 0.6
T[A["exploit"], S["initial"],    S["detected"]]   = 0.4
T[A["exploit"], S["probing"],    S["compromised"]]= 0.7
T[A["exploit"], S["probing"],    S["detected"]]   = 0.3
T[A["exploit"], S["compromised"],S["compromised"]]= 1.0
T[A["exploit"], S["detected"],   S["detected"]]   = 1.0

# ---------------------------------------------------------------------------
# 観測確率 Z[a, s', o] = P(o | s', a)
# ---------------------------------------------------------------------------

Z = np.zeros((na, ns, no))

# initial 状態では常に none
for a in range(na):
    Z[a, S["initial"], O["none"]] = 1.0

# probing 状態
Z[A["ssh_scan"],  S["probing"], O["service_found"]] = 0.8
Z[A["ssh_scan"],  S["probing"], O["none"]]          = 0.2
Z[A["vuln_scan"], S["probing"], O["vuln_found"]]    = 0.7
Z[A["vuln_scan"], S["probing"], O["service_found"]] = 0.3
Z[A["ssh_login"], S["probing"], O["none"]]          = 0.5
Z[A["ssh_login"], S["probing"], O["alert"]]         = 0.5
Z[A["exploit"],   S["probing"], O["none"]]          = 0.4
Z[A["exploit"],   S["probing"], O["alert"]]         = 0.6
Z[A["wait"],      S["probing"], O["none"]]          = 1.0

# compromised 状態
for a in range(na):
    Z[a, S["compromised"], O["auth_success"]] = 0.9
    Z[a, S["compromised"], O["none"]]         = 0.1

# detected 状態
for a in range(na):
    Z[a, S["detected"], O["alert"]] = 0.95
    Z[a, S["detected"], O["none"]]  = 0.05

# ---------------------------------------------------------------------------
# 報酬関数 R[s, a]
# ---------------------------------------------------------------------------

R = np.zeros((ns, na))
R[S["compromised"], :] = +200.0   # 侵入成功状態（吸収）
R[S["detected"],    :] = -150.0   # 検知状態（吸収）
R[S["probing"],  A["exploit"]]    = +50.0   # probing で exploit を促進
R[S["probing"],  A["ssh_login"]]  = +20.0   # probing で ssh_login を促進
R[S["initial"],  A["wait"]]       = -1.0
R[S["probing"],  A["wait"]]       = -1.0
R[S["initial"],  A["ssh_scan"]]   = +5.0    # 偵察は小さなプラス報酬

# ---------------------------------------------------------------------------
# コアアルゴリズム
# ---------------------------------------------------------------------------

def belief_update(belief: np.ndarray, action: int, obs: int) -> np.ndarray:
    """
    ベイズ更新による信念状態の更新

    b'(s') = η × Z(o | s', a) × Σ_s T(s' | s, a) × b(s)

    Parameters
    ----------
    belief : 現在の信念分布（長さ ns の確率ベクトル）
    action : 実行した行動のインデックス
    obs    : 得られた観測のインデックス

    Returns
    -------
    更新後の信念分布（正規化済み）
    """
    predicted = T[action].T @ belief          # Σ_s T(s'|s,a) × b(s)
    updated   = Z[action, :, obs] * predicted # Z(o|s',a) × predicted
    norm = updated.sum()
    if norm < 1e-10:
        return belief.copy()                  # 観測が起きえない場合は更新しない
    return updated / norm


# ---------------------------------------------------------------------------
# 方策
# ---------------------------------------------------------------------------

def pomdp_policy(belief: np.ndarray) -> int:
    """
    信念状態の閾値に基づく POMDP 方策（研究本体の簡易版）

    - detected への信念が高ければ wait（リスク回避）
    - compromised への信念が高ければ wait（侵入後に静観して報酬獲得）
    - probing への信念が高ければ exploit（侵入完了を狙う）
    - probing への信念が中程度であれば ssh_login
    - それ以外は ssh_scan で情報収集
    """
    b = {s: belief[i] for s, i in S.items()}
    if b["detected"] > 0.3:
        return A["wait"]
    if b["compromised"] > 0.5:
        return A["wait"]
    if b["probing"] > 0.4:
        return A["exploit"]
    if b["probing"] > 0.15:
        return A["ssh_login"]
    return A["ssh_scan"]


def greedy_policy(belief: np.ndarray) -> int:
    """期待報酬最大化のみを基準にする貪欲方策（比較用）"""
    expected = [float(belief @ R[:, a]) for a in range(na)]
    return int(np.argmax(expected))


def random_policy(belief: np.ndarray) -> int:
    """ランダム方策（比較用）"""
    return int(np.random.randint(na))


# ---------------------------------------------------------------------------
# エピソード実行
# ---------------------------------------------------------------------------

def run_episode(
    policy_fn: Callable,
    initial_belief: np.ndarray,
    initial_true_state: int,
    max_steps: int = 15,
) -> Tuple[float, List[dict]]:
    """
    1 エピソードをシミュレーション実行

    Parameters
    ----------
    policy_fn          : 方策関数（belief → action index）
    initial_belief     : 初期信念分布
    initial_true_state : 真の初期状態（デモ用に外部から与える）
    max_steps          : 最大ステップ数

    Returns
    -------
    (累積報酬, ステップごとの履歴)
    """
    belief = initial_belief.copy()
    true_state = initial_true_state
    total_reward = 0.0
    history: List[dict] = []

    for step in range(max_steps):
        action = policy_fn(belief)
        reward = float(R[true_state, action])
        total_reward += reward

        # 真の状態遷移
        next_state_probs = T[action, true_state]
        next_state = int(np.random.choice(ns, p=next_state_probs))

        # 観測のサンプリング
        obs_probs = Z[action, next_state]
        obs = int(np.random.choice(no, p=obs_probs))

        history.append({
            "step":   step + 1,
            "belief": {s: round(float(belief[i]), 3) for s, i in S.items()},
            "action": ACTIONS[action],
            "obs":    OBSERVATIONS[obs],
            "reward": reward,
        })

        # 信念更新
        belief = belief_update(belief, action, obs)
        true_state = next_state

        # 吸収状態（終了条件）
        if true_state in (S["compromised"], S["detected"]):
            # 最終ステップの信念も記録
            final_reward = float(R[true_state, A["wait"]])
            total_reward += final_reward
            history.append({
                "step":   step + 2,
                "belief": {s: round(float(belief[i]), 3) for s, i in S.items()},
                "action": "（終了）",
                "obs":    f"→ {STATES[true_state]}",
                "reward": final_reward,
            })
            break

    return total_reward, history


# ---------------------------------------------------------------------------
# 表示
# ---------------------------------------------------------------------------

def print_episode(history: List[dict], total_reward: float, policy_name: str) -> None:
    W = 70
    print(f"\n{'='*W}")
    print(f"  方策: {policy_name}")
    print(f"{'='*W}")
    header = f"{'Step':<5} {'行動':<14} {'観測':<16} {'報酬':>6}  信念分布 (in/pr/co/de)"
    print(header)
    print("-" * W)
    for h in history:
        b = h["belief"]
        bstr = f"{b['initial']:.2f}/{b['probing']:.2f}/{b['compromised']:.2f}/{b['detected']:.2f}"
        print(f"  {h['step']:<4} {h['action']:<14} {h['obs']:<16} {h['reward']:>+6.0f}  {bstr}")
    print("-" * W)
    print(f"  累積報酬: {total_reward:+.1f}")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  POMDP 信念状態更新デモ")
    print("  単一ホスト侵入シナリオ（概念実装 / ネットワーク接続なし）")
    print("=" * 70)
    print()
    print("  状態空間 : initial / probing / compromised / detected")
    print("  行動空間 : wait / ssh_scan / vuln_scan / ssh_login / exploit")
    print("  初期信念 : initial=0.85, detected=0.10, probing=0.05")
    print()
    print("  信念更新式:")
    print("    b'(s') = η × Z(o|s',a) × Σ_s T(s'|s,a) × b(s)")

    # 初期設定（研究と同じ初期信念）
    initial_belief = np.array([0.85, 0.05, 0.0, 0.10])

    # ---- 代表エピソード（1回）の詳細表示 ----
    print("\n【代表エピソードの信念更新過程】(seed=0)")
    for name, policy in [
        ("POMDP（閾値方策）", pomdp_policy),
        ("Greedy（期待報酬最大化）", greedy_policy),
    ]:
        np.random.seed(0)
        reward, history = run_episode(policy, initial_belief, S["initial"])
        print_episode(history, reward, name)

    # ---- 多数試行での平均累積報酬比較 ----
    N_TRIALS = 200
    print(f"\n【{N_TRIALS} エピソード平均累積報酬の比較】")
    print("=" * 70)

    for name, policy in [
        ("POMDP（閾値方策）", pomdp_policy),
        ("Greedy（期待報酬最大化）", greedy_policy),
        ("Random（ランダム）", random_policy),
    ]:
        rewards = []
        for seed in range(N_TRIALS):
            np.random.seed(seed)
            r, _ = run_episode(policy, initial_belief, S["initial"])
            rewards.append(r)
        avg = float(np.mean(rewards))
        bar = "+" * max(0, int(avg / 10)) if avg > 0 else "-" * max(0, int(-avg / 10))
        print(f"  {name:<28} 平均 {avg:>+8.1f}  {bar}")

    print()
    print("  ※ 研究本体（非公開）では4状態・5行動・7観測のフルモデルで")
    print("     実ネットワーク（VirtualBox/GNS3）上の実験を実施。")
    print("     POMDP が唯一正の累積報酬（+340.73）を達成。")
