"""Build a self-contained HTML report from the experiment CSV files."""

from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path

from mnist_data import PROJECT_ROOT


RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_FILES = [
    ("MLP（最佳 Dropout）", "mlp_metrics.csv", "mlp_confusion_matrix.csv", 101_770),
    ("早期 S/C CNN（1980 思路）", "early_cnn_metrics.csv", "early_cnn_confusion_matrix.csv", 5_994),
    ("现代两层 CNN（32/64 通道）", "modern_cnn_metrics.csv", "modern_cnn_confusion_matrix.csv", 421_642),
    ("LeNet-5", "lenet_metrics.csv", "lenet_confusion_matrix.csv", 61_706),
    ("线性 SVM", "svm_metrics.csv", "svm_confusion_matrix.csv", None),
    ("随机森林", "random_forest_metrics.csv", "random_forest_confusion_matrix.csv", None),
]


def read_first_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        return next(csv.DictReader(csv_file), None)


def read_matrix(path: Path) -> list[list[int]] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.reader(csv_file))
    if len(rows) != 11:
        return None
    try:
        return [[int(value) for value in row[1:]] for row in rows[1:]]
    except ValueError:
        return None


def model_details(row: dict) -> str:
    details = []
    if row.get("dropout_rate"):
        details.append(f"Dropout {row['dropout_rate']}")
    if row.get("c_value"):
        details.append(f"C={row['c_value']}")
    if row.get("n_estimators"):
        details.append(f"{row['n_estimators']} 棵树")
    if row.get("max_features"):
        details.append(f"max_features={row['max_features']}")
    if row.get("epochs"):
        details.append(f"{row['epochs']} epochs")
    if row.get("train_samples"):
        details.append(f"{int(row['train_samples']):,} 张训练图")
    return " · ".join(details) or "—"


def build_report(output_path: Path) -> None:
    models = []
    for display_name, metrics_name, matrix_name, parameter_count in MODEL_FILES:
        row = read_first_row(RESULTS_DIR / metrics_name)
        if not row or not row.get("test_accuracy"):
            continue
        try:
            test_accuracy = float(row["test_accuracy"])
        except (TypeError, ValueError):
            continue
        validation_value = row.get("best_validation_accuracy") or row.get(
            "validation_accuracy", ""
        )
        try:
            validation_accuracy = float(validation_value)
        except (TypeError, ValueError):
            validation_accuracy = None
        models.append(
            {
                "name": display_name,
                "test": test_accuracy,
                "validation": validation_accuracy,
                "details": model_details(row),
                "parameters": parameter_count,
                "matrix": read_matrix(RESULTS_DIR / matrix_name),
            }
        )

    if not models:
        raise FileNotFoundError(
            "没有找到可用的最终测试结果。请先运行模型训练脚本。"
        )

    models.sort(key=lambda item: item["test"], reverse=True)
    best = models[0]
    average = sum(item["test"] for item in models) / len(models)
    table_lines = []
    for index, item in enumerate(models, start=1):
        validation = (
            "—"
            if item["validation"] is None
            else f"{item['validation']:.2f}%"
        )
        parameters = (
            "—" if item["parameters"] is None else f"{item['parameters']:,}"
        )
        table_lines.append(
            "<tr>"
            f"<td><span class='rank'>{index:02d}</span> {html.escape(item['name'])}</td>"
            f"<td>{validation}</td>"
            f"<td><strong>{item['test']:.2f}%</strong></td>"
            f"<td><div class='bar-track'><div class='bar' style='width:{item['test']:.2f}%'></div></div></td>"
            f"<td>{parameters}</td>"
            f"<td class='muted'>{html.escape(item['details'])}</td>"
            "</tr>"
        )
    table_rows = "\n".join(table_lines)
    model_data = json.dumps(models, ensure_ascii=False).replace("</", "<\\/")
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    report = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MNIST 模型实验报告</title>
<style>
:root {{ color-scheme: light; --ink:#17243a; --muted:#66758b; --line:#e5ebf3; --blue:#4f6bed; --teal:#15a995; --bg:#f4f7fb; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 "Segoe UI","Microsoft YaHei",sans-serif; }}
.wrap {{ width:min(1120px, calc(100% - 36px)); margin:36px auto 60px; }}
.hero {{ padding:34px 38px; border-radius:22px; color:white; background:linear-gradient(120deg,#192e55,#3958b5 62%,#7867dc); box-shadow:0 14px 34px #253f7a25; }}
.eyebrow {{ opacity:.74; text-transform:uppercase; letter-spacing:.12em; font-size:12px; }}
h1 {{ margin:8px 0 6px; font-size:clamp(27px,4vw,40px); }}
.hero p {{ max-width:740px; margin:0; color:#e0e7ff; }}
.stamp {{ margin-top:20px; color:#c6d1fa; font-size:13px; }}
.cards {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:20px 0; }}
.card,.panel {{ background:white; border:1px solid var(--line); border-radius:16px; box-shadow:0 6px 18px #213a6410; }}
.card {{ padding:20px 22px; }}
.card-label {{ color:var(--muted); font-size:13px; }}
.card-value {{ display:block; margin-top:5px; font-size:27px; font-weight:750; }}
.card-note {{ color:var(--muted); font-size:12px; }}
.panel {{ padding:24px; margin:18px 0; }}
.panel h2 {{ margin:0 0 5px; font-size:20px; }}
.sub {{ margin:0 0 18px; color:var(--muted); font-size:13px; }}
.table-scroll {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ padding:13px 12px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
th {{ color:var(--muted); font-size:12px; font-weight:600; }}
tbody tr:last-child td {{ border-bottom:0; }}
.rank {{ display:inline-grid; place-items:center; width:25px; height:25px; margin-right:7px; border-radius:8px; background:#edf1ff; color:#4b60ca; font-size:11px; font-weight:700; }}
.bar-track {{ width:140px; height:9px; overflow:hidden; border-radius:99px; background:#edf1f7; }}
.bar {{ height:100%; border-radius:99px; background:linear-gradient(90deg,#4e70ec,#17b6a1); }}
.muted {{ color:var(--muted); font-size:13px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin:16px 0; }}
select {{ padding:9px 12px; border:1px solid var(--line); border-radius:9px; background:#fff; color:var(--ink); font:inherit; }}
.heatmap-wrap {{ overflow:auto; }}
.heatmap {{ width:auto; min-width:620px; border-collapse:separate; border-spacing:3px; }}
.heatmap th,.heatmap td {{ width:43px; min-width:43px; height:39px; padding:3px; border:0; border-radius:6px; text-align:center; font-size:12px; }}
.heatmap th {{ background:#f2f5fa; color:#54637a; }}
.legend {{ display:flex; align-items:center; gap:8px; color:var(--muted); font-size:12px; }}
.swatch {{ width:130px; height:9px; border-radius:8px; background:linear-gradient(90deg,#f1f5ff,#506bea); }}
.empty {{ padding:28px; color:var(--muted); text-align:center; }}
.note {{ margin-top:16px; padding:15px 17px; border-left:4px solid #6379e8; border-radius:8px; background:#f3f5ff; color:#43516b; font-size:13px; }}
.foot {{ color:var(--muted); text-align:center; font-size:12px; }}
@media(max-width:700px) {{ .cards {{ grid-template-columns:1fr; }} .hero {{ padding:27px 24px; }} .panel {{ padding:18px; }} }}
</style>
</head>
<body><main class="wrap">
<header class="hero"><div class="eyebrow">Machine Learning · MNIST</div><h1>手写数字识别实验报告</h1>
<p>在统一的 MNIST 测试集上比较 MLP、早期 S/C CNN、现代两层 CNN、LeNet-5、线性 SVM 与随机森林。所有准确率均来自各模型脚本保存的测试集结果。</p>
<div class="stamp">生成时间：{html.escape(generated)} · 数据集：MNIST · 测试图片：10,000 张</div></header>
<section class="cards">
<article class="card"><span class="card-label">当前最佳模型</span><strong class="card-value">{html.escape(best['name'])}</strong><span class="card-note">测试准确率 {best['test']:.2f}%</span></article>
<article class="card"><span class="card-label">模型数量</span><strong class="card-value">{len(models)}</strong><span class="card-note">已完成最终测试评估</span></article>
<article class="card"><span class="card-label">平均测试准确率</span><strong class="card-value">{average:.2f}%</strong><span class="card-note">当前报告内模型的算术平均</span></article>
</section>
<section class="panel"><h2>模型表现总览</h2><p class="sub">按测试集准确率排序。验证准确率用于选择配置或训练轮次；测试准确率用于最终比较。</p>
<div class="table-scroll"><table><thead><tr><th>模型</th><th>验证准确率</th><th>测试准确率</th><th>测试准确率图示</th><th>可训练参数</th><th>配置</th></tr></thead><tbody>{table_rows}</tbody></table></div></section>
<section class="panel"><h2>三个卷积架构如何区分？</h2>
<p class="sub">年代和网络容量是两条不同的比较轴；本报告展示的是这些具体实现的结果。</p>
<div class="note"><strong>早期 S/C CNN：</strong>借鉴 Fukushima 1980 年 Neocognitron 交替的特征检测层（S）和汇聚层（C），约 5,994 个参数。原始方法依靠无监督自组织；此教学版使用 PyTorch 反向传播、Adam 和 MNIST 标签训练，因此属于历史架构启发的适配实现。<a href="https://doi.org/10.1007/BF00344251">原始论文</a>。</div>
<div class="note"><strong>LeNet-5：</strong>约 61,706 个参数，使用 6/16 通道、Tanh 和平均池化。<strong>现代两层 CNN：</strong>约 421,642 个参数，使用 32/64 通道、ReLU、最大池化和 Dropout。三者模型容量不同；测试准确率展示本次配置下的表现，不能单独用于判断某个年代的架构普遍优劣。</div></section>
<section class="panel"><h2>测试集混淆矩阵</h2><p class="sub">行代表真实数字，列代表模型预测数字。对角线越大，代表该数字识别正确的数量越多。</p>
<div class="controls"><label for="modelSelect">查看模型：</label><select id="modelSelect"></select><div class="legend"><span>数量少</span><span class="swatch"></span><span>数量多</span></div></div>
<div id="matrixArea" class="heatmap-wrap"></div></section>
<p class="foot">本报告为离线 HTML，无需联网。详细训练过程和原始指标保存在同目录 CSV 文件中。</p>
</main>
<script>
const models = {model_data};
const selector = document.getElementById('modelSelect');
const area = document.getElementById('matrixArea');
models.forEach((model, index) => {{ const option=document.createElement('option'); option.value=index; option.textContent=model.name; selector.appendChild(option); }});
function showMatrix() {{
  const model=models[Number(selector.value)];
  if (!model.matrix) {{ area.innerHTML='<div class="empty">该模型没有找到混淆矩阵 CSV 文件。</div>'; return; }}
  const matrix=model.matrix; const max=Math.max(...matrix.flat(),1);
  let out='<table class="heatmap"><thead><tr><th>真实\\预测</th>'+Array.from({{length:10}},(_,i)=>'<th>'+i+'</th>').join('')+'</tr></thead><tbody>';
  matrix.forEach((row,i)=>{{out+='<tr><th>'+i+'</th>'; row.forEach(value=>{{const intensity=value/max; const alpha=(0.08+intensity*0.82).toFixed(3); const color=intensity>0.48?'#fff':'#243450'; out+='<td title="'+value+' 张" style="background:rgba(79,107,237,'+alpha+');color:'+color+'">'+value+'</td>';}}); out+='</tr>';}});
  area.innerHTML=out+'</tbody></table>';
}}
selector.addEventListener('change',showMatrix); showMatrix();
</script></body></html>'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"HTML report saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MNIST HTML results report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "index.html",
        help="Output HTML path (default: results/index.html).",
    )
    args = parser.parse_args()
    build_report(args.output.resolve())


if __name__ == "__main__":
    main()
