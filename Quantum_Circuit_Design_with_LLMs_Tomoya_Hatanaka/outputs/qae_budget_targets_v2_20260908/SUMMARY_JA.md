# 目標validation fidelityに達する最小予算の探索 — 日本語要約

日付: 2026-09-08 (JST)。protocol: `docs/research/QAE_BUDGET_TARGET_PROTOCOL.md`
(新規セルの候補を1つも生成する前に凍結・commit済み)。

## 1. 何を決めたか

指標は **validation trash fidelity** であり、reconstruction fidelityでも
test fidelityでもない。seed `s`、候補index `k` に対し
`F_best(s,k) = max_{i<=k} F_val(s,i)` を、**予算Bとして設定された**runの中で計算する。
`F_target ∈ {0.95, 0.99}` 以上に達したseedが、同じ12 paired seeds(0〜11)中
**10以上**なら合格。等号は到達とみなす。平均値による代用は不可。全12 seedを報告し、
seedの除外・引き直し・置換は行わない。

予算Bは「seedあたりに学習・評価する候補回路数」と定義する。token数でも
API呼出し数でも実行時間でもない。

### 前回資料からの意図的な変更点

前回の最終スライドは「厳密な最小値の探索が高コストな場合はB_minの区間を報告する」
としていた。本研究は**区間を報告しない**。設定予算Bごとに初期探索数・要求候補数・
改善horizonがすべて変わるため、各Bは別々の探索policyであり、あるBでの失敗と
より大きいBでの成功は、単調性を仮定しない限り最小値を挟み込まない。
代わりに **最小検証済み合格予算 / 検証済み失敗予算 / 未検証予算** の3分類で報告する。

## 2. 既存解析の再現(API呼出しなし)

source commit `e846c27fc21d06aa20e026590fcc06429f8bcfb4` の
`outputs/qae_robustness/` に対するvalidation限定の再解析を再実行し、
既報の監査結果を**byte単位で一致**して再現した(未改変scriptの実行結果と
出力ツリー全体を`diff -r`で照合。差異はscript自身の自己コピーのみで、これは
opt-in flagを追加したため)。

7条件・4手法の候補2,880件・選択336件について、seedの完全性(0〜11)、候補順序の
連続性、validation fidelityの有限性と値域、選択行の重複なし、各軌跡の
validation最大値と選択結果の一致を検査し、すべて通過した。test列は入力
allowlistで取り込み時に除外し、**test集計である旧 `anytime_mean.csv` は
予算選択に一切使用していない**。

依頼文中の照合値はすべて一致した。4-qubit Ising基準・目標0.95で
Open/Closedは B=4 で 0/12・8/12、B=8 で 10/12・12/12、B=16 で 12/12・11/12。
XXZ B=8 の Closed は 9/12。0.99ではどのセルも合格しない。

再利用した条件は対照であり、追加実験としては数えていない。

## 3. 今回実行した実験

事前登録した優先順位どおり2セル。

**E-A. 4-qubit Ising・基準model・B=6・全4手法・seeds 0〜11・Closedは3+3。**
基準条件(4 qubits / Ising / B=8 / 基準model)をanchorとし、変更factorは予算のみ。
LLM-Openは新規に6候補poolを生成した(B=8 poolの先頭6件ではない)。
Random/Greedyも同じB=6で実行し、対照を同予算に揃えた。

**E-B. 4-qubit XXZ・基準model・B=10・LLM-Closedのみ・seeds 0〜11・5+5。**
XXZ B=8条件をanchorとする。XXZ B=8自体が基準からの正当な1-factor条件である。
Closedのみを実行したため、**このセルには同予算の他手法比較は存在せず、報告もしない**。

旧one-factor検証器は無効化も弱体化もしていない。任意anchorを指定できるように
しただけで、既定anchorは不変、旧条件はすべて基準に対して検証を通り、第2anchorは
それ自体が基準からの清潔な1-factor条件であることを要求し、XXZ B=10 セルは
Ising基準に対して検査すれば依然として棄却される。新規2セルのmanifestの
frozen blockは基準とbyte一致する。

## 4. 結果 — 目標0.95の到達seed数

**P** は10/12規則を満たすセル。表中の "TFIM" は横磁場Ising鎖(slideでは Ising chain と表記)。
"not run" はそのセルでその手法を実行していないという意味であり、到達0件とは異なる。

| Condition | Configured B | Source | Random | Greedy | LLM-Open | LLM-Closed |
|---|---:|---|---:|---:|---:|---:|
| 4-qubit TFIM, reference model | 4 | re-analysed | 0/12 | 0/12 | 0/12 | 8/12 |
| 4-qubit TFIM, reference model | 6 | measured now | 0/12 | 2/12 | 12/12 **P** | 8/12 |
| 4-qubit TFIM, reference model | 8 | re-analysed | 1/12 | 1/12 | 10/12 **P** | 12/12 **P** |
| 4-qubit TFIM, reference model | 16 | re-analysed | 2/12 | 3/12 | 12/12 **P** | 11/12 **P** |
| 4-qubit XXZ, reference model | 8 | re-analysed | 2/12 | 3/12 | 0/12 | 9/12 |
| 4-qubit XXZ, reference model | 10 | measured now | not run | not run | not run | 5/12 |
| 6-qubit TFIM, reference model | 8 | re-analysed | 0/12 | 0/12 | 2/12 | 0/12 |
| 8-qubit TFIM, reference model | 8 | re-analysed | 0/12 | 0/12 | 0/12 | 0/12 |
| 4-qubit TFIM, alternative model | 8 | re-analysed | 1/12 | 1/12 | 7/12 | 5/12 |

## 5. 結果 — 目標0.99の到達seed数

| Condition | Configured B | Source | Random | Greedy | LLM-Open | LLM-Closed |
|---|---:|---|---:|---:|---:|---:|
| 4-qubit TFIM, reference model | 4 | re-analysed | 0/12 | 0/12 | 0/12 | 1/12 |
| 4-qubit TFIM, reference model | 6 | measured now | 0/12 | 0/12 | 0/12 | 0/12 |
| 4-qubit TFIM, reference model | 8 | re-analysed | 0/12 | 0/12 | 0/12 | 0/12 |
| 4-qubit TFIM, reference model | 16 | re-analysed | 0/12 | 0/12 | 0/12 | 0/12 |
| 4-qubit XXZ, reference model | 8 | re-analysed | 0/12 | 1/12 | 0/12 | 0/12 |
| 4-qubit XXZ, reference model | 10 | measured now | not run | not run | not run | 0/12 |
| 6-qubit TFIM, reference model | 8 | re-analysed | 0/12 | 0/12 | 0/12 | 0/12 |
| 8-qubit TFIM, reference model | 8 | re-analysed | 0/12 | 0/12 | 0/12 | 0/12 |
| 4-qubit TFIM, alternative model | 8 | re-analysed | 0/12 | 0/12 | 0/12 | 0/12 |

## 6. 新規2セルが示したこと

**Ising鎖では、open loopの方がclosed loopより少ない予算で足りる。**
B=6でOpenは 12/12 で合格、Closedは 8/12 で不合格。同じ予算・同じseedで
非意味的探索は遠く及ばない(Random 0/12、Greedy 2/12)。
実際に実行した予算の中では、**最小検証済み合格予算はOpenが6、Closedが8**。B=4は両者とも
検証済みの失敗である。

最も素直な解釈は構造的なものである。B=6のClosedは予算の半分をわずか3件の意味的初期探索に、残り半分を
3回のfeedback再設計に費やすのに対し、Openは6件すべてを1回の意味的batchに使う。予算が
小さいとき、二分割による損失がfeedbackの利得を上回っているように見える。ただし本研究で
変えたのは予算のみであり、これは観測された分割構造の解釈であって、原因を単離した結果ではない。B=6の到達曲線はこれを直接示しており、
Closedは改善phaseで5/12→8/12まで上がるが、合格線には届かない。

**予算を増やしてもXXZセルは救済されず、到達seed数はむしろ減った。**
XXZのClosedはB=8で 9/12(規則にseed 1つ足りない)、B=10で 5/12。
これは実測値としてそのまま報告する。ただしfidelityの崩壊ではない。最終best-so-far
validationの平均は 0.9588 → 0.9511(-0.0076)しか動いていない。XXZでは
分布全体が目標の数千分の1以内に密集しており(2 run合計 24 seed結果のうち
17 件が0.95から0.01以内)、到達seed数は極めて敏感な統計量になっている。
per-seed marginの図を参照。

**到達seed数は予算に対して単調ではない(実測)。**
Ising ladderの目標0.95で、Openは 0, 12, 10, 12、
Closedは 8, 8, 12, 11。XXZ anchorのClosedは
9 → 5。いずれも別policyの独立実行であり、run間の確率的変動は実在する。
これが、補間した最小値や挟み込み区間ではなく「検証済み合格 / 検証済み失敗 / 未検証」で
報告する具体的な理由である。データは、そうした主張が必要とする単調性を支持しない。

**新規2セルとも目標0.99は動かさない。** どちらも0/12で、既存の全条件と同じである。
その目標において探索・学習長・回路容量のどれが律速かは、ここからは特定できない。

## 7. run内の初到達index(新規セル)

| Condition | Configured B | Method | First k with 10/12 within the run |
|---|---:|---|---:|
| target_tfim_b6 | 6 | Random | not reached |
| target_tfim_b6 | 6 | Greedy | not reached |
| target_tfim_b6 | 6 | LLM-Open | 2 |
| target_tfim_b6 | 6 | LLM-Closed | not reached |
| target_xxz_b10 | 10 | LLM-Closed | not reached |

これは**実行済みrunのprefix**の記述であり、独立に実行された予算ではない。
候補poolは一括で要求・課金されており、設定予算を小さくすれば初期探索数・
要求候補数・改善horizonが変わる。長さkのprefixを「B=k実験」と呼ぶことはできず、
既に発生した生成費用も消えない。

Closedのrun前半は score を見ない意味的初期探索であり、feedbackによる再設計は
その後に始まる。この境界より前の到達をfeedback効果として説明することはできない。
またOpenのpoolはseed間で共有されているため、12 seedの成功は「1つのpoolに対する
12 seed」であって「12回の独立生成の成功率」ではない。

## 8. 生成妥当性

invalid / repaired / fallback は区別して記録している。ゲート契約に反する提案は
有限回の修復を試み、それでも失敗した場合はflag付きのrandom fallbackに置換される。
**random fallbackも候補予算を消費し、主解析に含めたままである。** 監査表の
fallback除外列は非因果的な補助診断にすぎない(既存ログのfilteringでは適応的探索の
履歴を再生成できない)。

| Cell | Method | Evaluated | Random fallbacks | With recorded invalid errors |
|---|---|---:|---:|---:|
| target_tfim_b6 | Random | 72 | 0 | 0 |
| target_tfim_b6 | Greedy | 72 | 0 | 0 |
| target_tfim_b6 | LLM-Open | 72 | 0 | 0 |
| target_tfim_b6 | LLM-Closed | 72 | 5 | 4 |
| target_xxz_b10 | LLM-Closed | 120 | 7 | 7 |

## 9. 費用・呼出し・時間

| Cell | Candidate evaluations | API calls | Repair/retry calls | Input tokens | Output tokens | Cost at list price (USD) |
|---|---:|---:|---:|---:|---:|---:|
| target_tfim_b6 | 288 | 57 | 8 | 43547 | 18187 | 0.1145 |
| target_xxz_b10 | 120 | 81 | 9 | 65969 | 28059 | 0.1757 |
| **total** | **408** | **138** | **17** | **109516** | **46246** | **0.2902** |

価格は基準modelのstandard tier公式価格(2026-09-08に
`https://developers.openai.com/api/docs/pricing` で確認)、token数は保存済みの
call recordから集計。今回の追加使用は **USD 0.2902**、model呼出し
138回(うち修復・再試行 17回)、input 109516 tokens、
output 46246 tokens、学習・評価した候補回路 408件。
実行時間は単一の数値ではなく `run_timeline.json` にsegment別に記録している。B=6は中断・再開を
経ており、両セルとも後に保存artefactから再生成しているため、どの1回の実行も「両セルのcold run」を
測っていない。唯一きれいなcold測定はXXZセルで、81 callsを227秒。

費用管理: `LLM_API_BUDGET_USD` に明示的な非ゼロ上限が設定されていない限り有料呼出しは
拒否され、各requestの**前**に残額を検査する。今回はユーザーが承認した上限 USD 2.00 の下で
実行した。B=6のrunは10/12 seed完了時点で一時的なDNS障害により中断したため、
**既に消費した分だけ上限を減額して再開**し、累積が2.00を超えないようにした。再開時は
保存済みの10 seedとpoolをdiskから読み直し、これらについて新規呼出しは発生していない
(設定予算・model・prompt・seed・初期化・dataが完全一致する結果のみ再利用)。

## 10. testの保護

本研究では**どの候補もtest setで評価していない**。runnerは凍結trainerに空のtest配列を
渡すため、学習済み回路がtest状態と縮約されることが構造的に起こらない。記録されたtest量が
有限値になった場合はguardがrunを失敗させ、出力表はtest列を持たないvalidation限定の
column allowlistで書かれる。予算・回路・解析方針はvalidationのみで選んだ。

過去の研究にはtest出力が存在するが、それらは歴史的な記述結果であり、新しい未使用の
確認集合ではない。本報告では一切報告していない。将来確認評価を行う場合は、別途事前登録し、
全選択を凍結した後にのみ参照し、結果を見て探索へ戻らないこと。

## 11. 検証済み / 失敗 / 未検証

**4-qubit Ising chain, reference model**, target 0.95:

- `Random` - smallest verified passing budget: none among those run; verified failing budgets: 4, 6, 8, 16; unverified budgets: 2, 10, 12, 14 (and every budget above 16).
- `Greedy` - smallest verified passing budget: none among those run; verified failing budgets: 4, 6, 8, 16; unverified budgets: 2, 10, 12, 14 (and every budget above 16).
- `LLM-Open` - smallest verified passing budget: **B = 6**; verified failing budgets: 4; unverified budgets: 2, 10, 12, 14 (and every budget above 16).
- `LLM-Closed` - smallest verified passing budget: **B = 8**; verified failing budgets: 4, 6; unverified budgets: 2, 10, 12, 14 (and every budget above 16).

**4-qubit XXZ chain, reference model**, target 0.95:

- `LLM-Closed` - smallest verified passing budget: none among those run; verified failing budgets: 8, 10; unverified budgets: 2, 4, 6, 12, 14, 16 (and every budget above 10).

At target 0.99 no method passes at any budget or condition tested, including the two newly executed cells. That target is unresolved, and the runs performed here do not identify whether search, training or circuit capacity is the binding constraint.

A verified passing budget is **not** evidence that a smaller untested budget fails, and a verified failing budget is not evidence that every smaller budget fails. Each budget was executed as its own policy.

## 12. 限界

12 paired seeds、ノイズなし状態ベクトルsimulation、回路容量規則は1つ
(qubitあたり3 rotation + 1 CNOT)、新規セルのmodel snapshotは1つ、XXZ anchorは1点。
各予算は別policyとして実行しており、Bに関する単調性は仮定せず二分探索も行っていない。
Openのpoolはseed間で共有。random fallbackは運用スコアに含まれるため、生成妥当性と
アーキテクチャ品質はすべてのLLM数値において分離されていない。探索不足・学習不足・
容量不足のいずれが律速かは本研究では分離できておらず、特に0.99の未到達については
どれが原因かの証拠を与えない。

## 13. 再現手順

英語版REPORTの Section 12 を参照。
