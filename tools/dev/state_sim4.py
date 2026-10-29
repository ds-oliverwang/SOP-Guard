"""Round 4: hand-picked example frames (what a user would pick by eye: hands out, contents visible), then the policy grid."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import state_sim as S
import state_sim2 as S2
import state_sim3 as S3

# 由 contact sheet 人工挑的格：未開＝蓋著的紅盒（開盒前／蓋上後）、空盒、已放防黴片＝黑卡＋白貼紙都在、已放鞋子＝鞋在盒內看得到
EX = [("未開", 185), ("未開", 200), ("未開", 745), ("未開", 755),
      ("空盒", 290), ("空盒", 295), ("空盒", 320),
      ("已放防黴片", 505), ("已放防黴片", 525), ("已放防黴片", 540),
      ("已放鞋子", 650), ("已放鞋子", 665), ("已放鞋子", 710),
      ("未開", 1330), ("未開", 1895), ("未開", 1905),
      ("空盒", 1470), ("空盒", 1475),
      ("已放防黴片", 1655), ("已放防黴片", 1665),
      ("已放鞋子", 1830), ("已放鞋子", 1845), ("已放鞋子", 1770),
      ("未開", 2535), ("未開", 2545),
      ("空盒", 2665), ("空盒", 2680),
      ("已放防黴片", 2850), ("已放防黴片", 2865),
      ("已放鞋子", 3005), ("已放鞋子", 3015), ("已放鞋子", 3035)]

if __name__ == "__main__":
    tmp = os.path.join(S.ROOT, "cache", "_sim_lib4"); gate = S2.gate_inner
    lib, used = S2.build_library(EX, tmp)
    print("##### hand-picked examples:", lib.counts(), lib.leave_one_out())
    print("   frames:", used)
    for kw in [dict(), dict(end_hold=1.0), dict(votes=3, min_votes=2, end_hold=1.0), dict(judge_within=10**9, end_hold=1.0),
               dict(gone_reset=False, end_hold=1.0), dict(every=10, end_hold=1.0)]:
        tl = S3.simulate(lib, gate, label="hand-picked " + str(kw), **kw)
        S.save_cache()
    S2.report_raw(tl, "hand-picked")
