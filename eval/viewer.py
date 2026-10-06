"""Local results viewer (dev tool, not part of the agent). Renders eval/results/*.json into one HTML page and opens it in the browser.

python eval/viewer.py            # build eval/results/viewer.html + open it
python eval/viewer.py --no-open  # build only
"""
import argparse
import json
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval/results"

PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Doctor-Agent 진료 기록</title>
<style>
:root {
  --bg: #f6f7f9; --card: #ffffff; --text: #1d2330; --muted: #6b7385; --line: #e3e6ec;
  --ask: #2f6fdb; --exam: #0f8a6a; --test: #9b5de5; --dx: #d9480f;
  --doctor-bg: #eef3fc; --patient-bg: #f3f4f6; --good: #0f8a6a; --mid: #c98a00; --bad: #d63939;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12151b; --card: #1b1f27; --text: #e6e9ef; --muted: #9aa3b5; --line: #2a303b;
    --doctor-bg: #1e2a40; --patient-bg: #232833; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #12151b; --card: #1b1f27; --text: #e6e9ef; --muted: #9aa3b5; --line: #2a303b;
  --doctor-bg: #1e2a40; --patient-bg: #232833; color-scheme: dark;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif; }
.wrap { max-width: 880px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: var(--muted); font-size: 13px; }
select { font: inherit; padding: 6px 10px; border-radius: 8px; border: 1px solid var(--line);
  background: var(--card); color: var(--text); max-width: 100%; }
.bar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 16px 0 20px; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-bottom: 24px; }
.stat { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; }
.stat .k { color: var(--muted); font-size: 12px; }
.stat .v { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; margin-bottom: 20px; overflow: hidden; }
.card > summary { list-style: none; cursor: pointer; padding: 14px 16px; display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; }
.card > summary::-webkit-details-marker { display: none; }
.case-id { font-weight: 700; }
.pill { font-size: 12px; padding: 2px 8px; border-radius: 99px; border: 1px solid var(--line); color: var(--muted); }
.pill.good { color: var(--good); border-color: var(--good); }
.pill.mid { color: var(--mid); border-color: var(--mid); }
.pill.bad { color: var(--bad); border-color: var(--bad); }
.body { padding: 0 16px 16px; border-top: 1px solid var(--line); }
.initial { margin: 14px 0; padding: 10px 12px; border-left: 3px solid var(--muted); color: var(--muted); }
.turn { margin: 12px 0; }
.doctor { background: var(--doctor-bg); border-radius: 12px 12px 12px 4px; padding: 10px 12px; margin-right: 12%; }
.why { margin-top: 6px; font-size: 13px; color: var(--muted); }
.why b { font-weight: 600; }
.patient { background: var(--patient-bg); border-radius: 12px 12px 4px 12px; padding: 10px 12px; margin: 6px 0 0 12%; }
.who { font-size: 12px; color: var(--muted); margin-bottom: 2px; display: flex; gap: 6px; align-items: center; }
.tag { font-size: 11px; font-weight: 700; color: #fff; padding: 1px 7px; border-radius: 6px; }
.t-ASK { background: var(--ask); } .t-EXAM { background: var(--exam); } .t-TEST { background: var(--test); } .t-DIAGNOSE { background: var(--dx); }
.result { margin-top: 16px; display: grid; gap: 6px; font-size: 14px; }
.result b { display: inline-block; min-width: 88px; color: var(--muted); font-weight: 500; }
.empty { color: var(--muted); padding: 40px 0; text-align: center; }
h2 { font-size: 16px; margin: 28px 0 10px; }
.cmp-wrap { overflow-x: auto; background: var(--card); border: 1px solid var(--line); border-radius: 12px; }
table.cmp { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
.cmp th, .cmp td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
.cmp th { color: var(--muted); font-weight: 500; }
.cmp tr:last-child td { border-bottom: 0; }
.cmp tr.sel td { background: var(--doctor-bg); }
.cmp tr[data-i] { cursor: pointer; }
details.sets { margin: -12px 0 20px; }
details.sets > summary { cursor: pointer; color: var(--muted); font-size: 13px; margin-bottom: 8px; }
.filters { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 0 0 14px; font-size: 13px; }
.filters select { padding: 4px 8px; font-size: 13px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>진료 기록 뷰어</h1>
  <div class="sub">로컬 평가 결과를 대화 형태로 보여줍니다</div>
  <h2>실험 비교</h2>
  <div class="cmp-wrap"><table class="cmp" id="cmp"></table></div>
  <div class="sub" style="margin-top:6px">행을 누르면 아래에 그 실행의 진료 기록이 나옵니다 · 안전성은 지침 기반 확인 목록 기준</div>
  <h2>진료 기록</h2>
  <div class="bar">
    <label for="run">실행 기록</label>
    <select id="run"></select>
    <span class="sub" id="models"></span>
  </div>
  <div class="stats" id="stats"></div>
  <details class="sets" id="sets-box"><summary>세트별 점수</summary><div class="cmp-wrap"><table class="cmp" id="sets"></table></div></details>
  <div class="filters">
    <label for="f-set">세트</label><select id="f-set"></select>
    <label for="f-ok">정답 여부</label>
    <select id="f-ok"><option value="">전체</option><option value="ok">정답</option><option value="bad">오답(부분점수 포함)</option></select>
    <span class="sub" id="f-count"></span>
  </div>
  <div id="cases"></div>
</div>
<script>
const RUNS = __DATA__;
const LABEL = { ASK: "문진", EXAM: "진찰", TEST: "검사", DIAGNOSE: "진단" };
const SCORE = { accuracy: "정확도", efficiency: "효율성", safety: "안전성", case_checks: "증례 체크" };
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const tone = v => v == null ? "" : v >= 0.8 ? "good" : v >= 0.5 ? "mid" : "bad";
const fmt = v => v == null ? "–" : Number(v).toFixed(2);

function stamp(name) {
  const m = name.match(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/);
  return m ? `${m[1]}.${m[2]}.${m[3]} ${m[4]}:${m[5]}:${m[6]}` : name;
}

const avgOf = (cs, k) => { const v = cs.map(c => (c.scores || {})[k]).filter(x => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
const isOk = c => ((c.scores || {}).accuracy ?? 0) >= 1;
const fSet = document.getElementById("f-set"), fOk = document.getElementById("f-ok");
let current = 0;

function renderSets(cases) {
  const groups = {};
  cases.forEach(c => { (groups[c.set || "–"] = groups[c.set || "–"] || []).push(c); });
  const names = Object.keys(groups).sort();
  const box = document.getElementById("sets-box");
  box.style.display = names.length > 1 || (names.length === 1 && names[0] !== "–") ? "" : "none";
  const cell = v => `<td style="color:var(--${tone(v) || "text"})">${fmt(v)}</td>`;
  document.getElementById("sets").innerHTML =
    `<tr><th>세트</th><th>증례</th><th>정확도</th><th>효율성</th><th>안전성</th><th>평균 턴</th><th>정답 수</th></tr>` +
    names.map(n => { const g = groups[n];
      return `<tr><td>${esc(n)}</td><td>${g.length}</td>${cell(avgOf(g, "accuracy"))}${cell(avgOf(g, "efficiency"))}${cell(avgOf(g, "safety"))}` +
        `<td>${(g.reduce((a, c) => a + c.n_turns, 0) / g.length).toFixed(1)}</td><td>${g.filter(isOk).length}</td></tr>`; }).join("");
  const keep = fSet.value;
  fSet.innerHTML = `<option value="">전체</option>` + names.map(n => `<option value="${esc(n)}">${esc(n)} (${groups[n].length})</option>`).join("");
  fSet.value = names.includes(keep) ? keep : "";
}

function render(i) {
  current = i;
  const run = RUNS[i].data;
  const all = run.cases || [];
  const meta = [run.doctor_model && `의사 모델: ${run.doctor_model}`, run.label && `라벨: ${run.label}`,
    run.prompt_version && `프롬프트: ${run.prompt_version}`, run.commit && `커밋: ${run.commit}`].filter(Boolean);
  document.getElementById("models").textContent = meta.join(" · ");
  renderSets(all);
  renderCases(all, run);
}

function renderCases(all, run) {
  const cases = all.filter(c => (!fSet.value || (c.set || "–") === fSet.value) &&
    (!fOk.value || (fOk.value === "ok") === isOk(c)));
  document.getElementById("f-count").textContent = `${cases.length} / ${all.length}개`;
  const avgTurns = cases.length ? (cases.reduce((a, c) => a + c.n_turns, 0) / cases.length).toFixed(1) : "–";
  document.getElementById("stats").innerHTML =
    Object.entries(run.avg || {}).map(([k, v]) => [k, cases.length === all.length ? v : avgOf(cases, k)]).map(([k, v]) =>
      `<div class="stat"><div class="k">평균 ${SCORE[k] || k}</div><div class="v" style="color:var(--${tone(v) || "text"})">${fmt(v)}</div></div>`).join("") +
    `<div class="stat"><div class="k">증례 수</div><div class="v">${cases.length}</div></div>` +
    `<div class="stat"><div class="k">평균 턴 수</div><div class="v">${avgTurns}</div></div>`;

  document.getElementById("cases").innerHTML = cases.length ? cases.map((c, idx) => {
    const s = c.scores || {};
    const pills = Object.entries(s).map(([k, v]) => `<span class="pill ${tone(v)}">${SCORE[k] || k} ${fmt(v)}</span>`).join("");
    const reviewAt = {};
    (c.reviews || []).forEach(r => { (reviewAt[r.turn] = reviewAt[r.turn] || []).push(r); });
    const reviewHtml = n => (reviewAt[n + 1] || []).map(r =>
      `<div class="why"><b>검토의</b> · ${r.verdict === "보류" ? "⏸ 보류" : "✅ 승인"} (제안 진단: ${esc(r.proposed)})` +
      `${r.issues && r.issues.length ? " — " + r.issues.map(esc).join("; ") : ""}` +
      `${r.final_diagnosis ? ` → 진단명 수정: ${esc(r.final_diagnosis)}` : ""}</div>`).join("");
    const safetyAt = {};
    (c.safety_log || []).forEach(e => { (safetyAt[e.turn] = safetyAt[e.turn] || []).push(e); });
    const LAYER = { grounding: "사실 확인", danger_gate: "위험 질환 관문", preconditions: "검사 전 확인",
                    confidence: "확신도", anchoring: "감별 넓히기", planner: "추천 행동", triage: "중증도", result_interp: "결과 판독", advisors: "보조 모듈",
                    subagent: "전문 자문" };
    const safetyHtml = n => (safetyAt[n + 1] || []).map(e => {
      let msg = e.error ? "오류: " + e.error
        : e.msg ? e.msg
        : e.layer === "grounding" ? `미확인 소견 ${e.findings_unverified || 0}개` + (e.unverified_examples && e.unverified_examples.length ? ` (${e.unverified_examples.join(", ")})` : "") + (e.ddx_removed ? `, 근거에서 뺀 항목 ${e.ddx_removed}개` : "")
        : e.layer === "danger_gate" ? (e.kind === "rule_out" ? `진단(${e.proposed}) 보류 → ${e.danger} 배제 먼저: ${e.why || ""}` : `참고: ${e.why || ""}`)
        : `${e.severity === "block" ? "차단" : "경고"}: ${e.requested} — ${e.why || ""}` + (e.replaced_with ? ` → ${e.replaced_with.content}` : "");
      return `<div class="why"><b>${LAYER[e.layer] || e.layer}</b> · ${esc(msg)}</div>`;
    }).join("");
    const turns = (c.turns || []).map((t, n) => `
      <div class="turn">
        <div class="doctor"><div class="who">의사 · ${n + 1}턴 <span class="tag t-${t.type}">${LABEL[t.type] || t.type}</span></div>${esc(t.content)}${t.reason ? `<div class="why"><b>근거</b> · ${esc(t.reason)}</div>` : ""}${reviewHtml(n)}${safetyHtml(n)}${t.ddx && t.ddx.length ? `<div class="why"><b>감별 후보</b> · ${t.ddx.map(d => typeof d === "object" ? `${esc(d.dx)}${d.p != null ? ` ${Math.round(d.p * 100)}%` : ""}` : esc(d)).join(", ")}</div>` : ""}</div>
        ${t.type === "DIAGNOSE" ? "" : `<div class="patient"><div class="who">${t.type === "ASK" ? "환자" : "결과"}</div>${esc(t.response)}</div>`}
      </div>`).join("");
    return `
      <details class="card" ${idx === 0 ? "open" : ""}>
        <summary><span class="case-id">${esc(c.case)}</span>${c.set ? `<span class="pill">${esc(c.set)}</span>` : ""}${c.persona ? `<span class="pill">${esc(c.persona)}</span>` : ""}<span class="pill">${c.n_turns}턴</span>${pills}<span class="sub">${c.sec ?? "–"}초</span></summary>
        <div class="body">
          ${c.initial ? `<div class="initial">처음 정보 · ${esc(c.initial)}</div>` : ""}
          ${turns}
          <div class="result">
            <div><b>에이전트 진단</b>${esc(c.diagnosis || "없음")}</div>
            <div><b>정답</b>${esc(c.answer)}</div>
            ${c.judge ? `<div><b>채점 이유</b>${esc(c.judge.reason)}</div>` : ""}
            ${c.findings && c.findings.length ? `<div><b>소견 장부</b>${["양성", "음성", "결과없음"].map(st => {
                const g = c.findings.filter(f => f.status === st).map(f => esc(f.item) + (f.detail ? ` (${esc(f.detail)})` : ""));
                return g.length ? `<span class="sub">${st}</span> ${g.join(", ")}` : "";
              }).filter(Boolean).join(" · ")}</div>` : ""}
            ${c.missed_checks && c.missed_checks.length ? `<div><b>빠뜨린 확인</b>${c.missed_checks.map(esc).join(", ")}</div>` : ""}
            ${c.ddx && c.ddx.length ? `<div><b>감별 후보</b>${c.ddx.map(d => esc(typeof d === "object" ? `${d.dx}${d.p != null ? ` (${fmt(d.p)})` : ""}` : d)).join(", ")}</div>` : ""}
          </div>
        </div>
      </details>`;
  }).join("") : `<div class="empty">이 실행에는 증례가 없습니다.</div>`;
}

const PERSONA = { standard: "보통 환자", mixed: "까다로운 환자(섞음)", vague: "모호한 환자", anxious: "불안한 환자",
  minimizer: "증상을 축소하는 환자", poor_historian: "기억이 흐린 환자" };
function label(r) {
  const d = r.data, n = (d.cases || []).length;
  const who = (d.label ? d.label + " · " : "") + (d.doctor_model || "–") + (d.persona ? ` · ${PERSONA[d.persona] || d.persona}` : "");
  return `${stamp(r.name)} · ${who} · ${n}개`;
}
function renderCmp(active) {
  const head = `<tr><th>시각</th><th>라벨</th><th>프롬프트</th><th>커밋</th><th>의사 모델</th><th>환자 유형</th><th>증례</th><th>정확도</th><th>효율성</th><th>안전성</th><th>평균 턴</th></tr>`;
  const rows = RUNS.map((r, i) => {
    const d = r.data, cs = d.cases || [], a = d.avg || {};
    const turns = cs.length ? (cs.reduce((x, c) => x + c.n_turns, 0) / cs.length).toFixed(1) : "–";
    const cell = v => `<td style="color:var(--${tone(v) || "text"})">${fmt(v)}</td>`;
    return `<tr data-i="${i}" class="${i == active ? "sel" : ""}"><td>${stamp(r.name)}</td><td>${esc(d.label || "–")}</td>` +
      `<td>${esc(d.prompt_version || "–")}</td><td>${esc(d.commit || "–")}</td><td>${esc(d.doctor_model || "–")}</td>` +
      `<td>${esc(d.persona ? (PERSONA[d.persona] || d.persona) : (cs[0] && cs[0].persona) || "–")}</td><td>${cs.length}</td>` +
      cell(a.accuracy) + cell(a.efficiency) + cell(a.safety) + `<td>${turns}</td></tr>`;
  }).join("");
  const t = document.getElementById("cmp");
  t.innerHTML = head + rows;
  t.querySelectorAll("tr[data-i]").forEach(tr => tr.onclick = () => { sel.value = tr.dataset.i; show(tr.dataset.i); });
}
function show(i) { render(i); renderCmp(i); }
const sel = document.getElementById("run");
if (!RUNS.length) {
  document.getElementById("cases").innerHTML = `<div class="empty">아직 결과가 없습니다. 먼저 <code>python eval/run_local.py</code>를 실행하세요.</div>`;
} else {
  sel.innerHTML = RUNS.map((r, i) => `<option value="${i}">${esc(label(r))}</option>`).join("");
  sel.onchange = () => show(sel.value);
  fSet.onchange = fOk.onchange = () => renderCases(RUNS[current].data.cases || [], RUNS[current].data);
  show(0);
}
</script>
</body>
</html>
"""


def is_dummy(data: dict) -> bool:
    """Scripted smoke-test runs (--doctor dummy): they only check that the code runs, their scores mean nothing."""
    return data.get("doctor_model") == "dummy"


def build(include_dummy: bool = False) -> Path:
    runs = []
    for p in sorted(RESULTS.glob("run_*.json"), reverse=True):  # newest first
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if include_dummy or not is_dummy(data):
            runs.append({"name": p.stem, "data": data})
    data = json.dumps(runs, ensure_ascii=False).replace("</", "<\\/")
    out = RESULTS / "viewer.html"
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace("__DATA__", data), encoding="utf-8")
    return out


def build_share(out: Path, include_dummy: bool = False) -> Path:
    """Page body for publishing as a hosted artifact: no document skeleton (the host adds it)."""
    html = build(include_dummy).read_text(encoding="utf-8")
    start = html.index("<title>")
    body = html[start:].replace("</head>\n<body>\n", "", 1)
    body = body.rsplit("</body>", 1)[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--share", metavar="PATH", help="also write a share-ready page (no html/head/body skeleton)")
    ap.add_argument("--include-dummy", action="store_true", help="also show scripted smoke-test runs (--doctor dummy)")
    args = ap.parse_args()
    out = build(args.include_dummy)
    print(f"viewer: {out}")
    if args.share:
        print(f"share page: {build_share(Path(args.share), args.include_dummy)}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
