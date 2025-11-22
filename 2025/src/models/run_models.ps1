param(
    [string]$Mode = "menu",
    [string]$Questions = "",
    [switch]$NoML
)

# ---------------------------------------------
# APMCM 2025 C 题模型统一启动脚本（PowerShell）
# 位置：2025/src/models/run_models.ps1
# 主要用于在 Windows 下一键运行数据预处理和 Q1-Q5 模型
# ---------------------------------------------

# 获取项目根目录和关键脚本路径
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcDir    = Join-Path $scriptDir ".." | Resolve-Path
$projectDir = Join-Path $srcDir ".." | Resolve-Path

$mainPy   = Join-Path $srcDir "main.py"
$prepPy   = Join-Path $srcDir "preprocessing\prepare_data.py"
${q1SoyPy} = Join-Path $srcDir "models\q1_soybeans.py"

function Invoke-ProjectPython {
    param(
        [string[]]$PyArgs
    )
    # 优先使用 uv run python，其次使用本机 python
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Host "[信息] 使用 uv run python 执行：" -ForegroundColor Cyan
        Write-Host "       " ($PyArgs -join " ") `n
        & uv run python @PyArgs
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        Write-Host "[信息] 未找到 uv，改用系统 python 执行：" -ForegroundColor Yellow
        Write-Host "       " ($PyArgs -join " ") `n
        & python @PyArgs
    } else {
        Write-Host "[错误] 未找到 uv 或 python，请先在系统中安装 Python/uv。" -ForegroundColor Red
        exit 1
    }
}

function Run-PrepareData {
    if (-not (Test-Path $prepPy)) {
        Write-Host "[警告] 未找到数据预处理脚本：$prepPy" -ForegroundColor Yellow
        return
    }
    Write-Host "========== 步骤 1：运行数据预处理脚本 ==========" -ForegroundColor Green
    Invoke-ProjectPython @($prepPy)
}

function Run-MainByQuestion {
    param(
        [string]$Q,
        [switch]$NoMLFlag
    )
    if (-not (Test-Path $mainPy)) {
        Write-Host "[警告] 未找到主运行脚本 main.py：$mainPy" -ForegroundColor Yellow
        Write-Host "        可以考虑直接运行单个模型文件（例如 q2_autos.py）。" -ForegroundColor Yellow
        return
    }
    $args = @($mainPy)
    if ($Q) {
        $args += @("--questions", $Q)
    }
    if ($NoMLFlag.IsPresent) {
        $args += "--no-ml"
    }
    if (-not $Q) {
        Write-Host "[信息] 将通过 main.py 运行默认配置。" -ForegroundColor Cyan
    } else {
        Write-Host "[信息] 将通过 main.py 运行问题：$Q" -ForegroundColor Cyan
    }
    if ($NoMLFlag.IsPresent) {
        Write-Host "[信息] 已启用 --no-ml（只运行经济计量/基准模型）。" -ForegroundColor Cyan
    }
    Invoke-ProjectPython $args
}

function Show-Menu {
    Write-Host "" 
    Write-Host "=============================================" -ForegroundColor Cyan
    Write-Host " APMCM 2025 C题 模型统一启动菜单 (run_models.ps1)" -ForegroundColor Cyan
    Write-Host " 项目根目录：$projectDir" -ForegroundColor DarkCyan
    Write-Host "=============================================" -ForegroundColor Cyan
    Write-Host " 1) 只运行数据预处理 (prepare_data.py)"
    Write-Host " 2) 运行 Q1 大豆贸易模型"
    Write-Host " 3) 运行 Q2 汽车贸易 + MARL + SAC + Transformer"
    Write-Host " 4) 运行 Q3 半导体贸易 + GNN + ML"
    Write-Host " 5) 运行 Q4 关税收入 Laffer 曲线 + ML"
    Write-Host " 6) 运行 Q5 宏观金融 VAR + ML + Transformer (PyTorch)"
    Write-Host " 7) 运行所有模型（不含 ML，适合快速回归）"
    Write-Host " 8) 运行所有模型（含 ML 与深度学习扩展）"
    Write-Host " 9) 仅运行 Q1 弹性回归修复 (q1_soybeans.py --fix-elasticity)"
    Write-Host " 0) 退出"
    Write-Host "=============================================" -ForegroundColor Cyan
    $choice = Read-Host "请输入选项编号"
    return $choice
}

# 主控制逻辑
if ($Mode -ne "menu" -and $Questions) {
    # 非交互模式：直接通过 main.py 按问题编号运行
    Run-PrepareData
    Run-MainByQuestion -Q $Questions -NoMLFlag:$NoML
    exit 0
}

# 交互菜单模式
while ($true) {
    $opt = Show-Menu
    switch ($opt) {
        "1" {
            Run-PrepareData
        }
        "2" {
            Run-PrepareData
            Run-MainByQuestion -Q "Q1" -NoMLFlag:$NoML
        }
        "3" {
            Run-PrepareData
            Run-MainByQuestion -Q "Q2" -NoMLFlag:$NoML
        }
        "4" {
            Run-PrepareData
            Run-MainByQuestion -Q "Q3" -NoMLFlag:$NoML
        }
        "5" {
            Run-PrepareData
            Run-MainByQuestion -Q "Q4" -NoMLFlag:$NoML
        }
        "6" {
            Run-PrepareData
            Run-MainByQuestion -Q "Q5" -NoMLFlag:$NoML
        }
        "7" {
            Run-PrepareData
            Write-Host "[信息] 以 --no-ml 模式运行所有问题 (Q1-Q5)。" -ForegroundColor Cyan
            Run-MainByQuestion -Q "Q1" -NoMLFlag
            Run-MainByQuestion -Q "Q2" -NoMLFlag
            Run-MainByQuestion -Q "Q3" -NoMLFlag
            Run-MainByQuestion -Q "Q4" -NoMLFlag
            Run-MainByQuestion -Q "Q5" -NoMLFlag
        }
        "8" {
            Run-PrepareData
            Write-Host "[信息] 运行所有问题 (Q1-Q5)，包含 ML/深度学习部分。" -ForegroundColor Cyan
            Run-MainByQuestion -Q "Q1"
            Run-MainByQuestion -Q "Q2"
            Run-MainByQuestion -Q "Q3"
            Run-MainByQuestion -Q "Q4"
            Run-MainByQuestion -Q "Q5"
        }
        "9" {
            # 仅运行 Q1 弹性回归修复，不跑完整 Q1 流水线
            if (-not (Test-Path $q1SoyPy)) {
                Write-Host "[警告] 未找到 Q1 弹性脚本：$q1SoyPy" -ForegroundColor Yellow
            } else {
                Write-Host "[信息] 仅运行 Q1 弹性回归修复 (q1_soybeans.py --fix-elasticity)" -ForegroundColor Cyan
                Invoke-ProjectPython @($q1SoyPy, "--fix-elasticity")
            }
        }
        "0" {
            Write-Host "已退出。" -ForegroundColor Green
            break
        }
        Default {
            Write-Host "无效选项，请重新输入。" -ForegroundColor Yellow
        }
    }
}
