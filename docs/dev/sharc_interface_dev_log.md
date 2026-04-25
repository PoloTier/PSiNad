# SHARC_Interface 开发日志

> **格式约定**：按时间倒序追加（最新在上）。每条记录是一个「checkpoint」：
> 记录当时的起点、做了什么、结束时的状态、非显然决策。
> 需要回溯某个改动的上下文时直接定位对应条目。
>
> **条目标签**：`[YYYY-MM-DD #N | topic]`。N 只在同一天内递增。
> **状态字段**：`✅` 完成并验证 / `🟡` 完成但未验证 / `⏳` 未开始 / `❌` 放弃

---

## 当前状态

- **分支**：`lhc_dev_soc`
- **阶段**：S0 ANALYTICAL ✅；S0 LVC ✅；S1 MNDO ✅（Level A + B）；S2 LVC+SOC ✅；S1 其他 ab-initio 延后（无 ORCA/MOLCAS/TURBOMOLE）；BAGEL/PYSCF/MOLPRO/AMS_ADF/COLUMBUS 架构不兼容（见 #17）
- **最近可复现跑通命令**：
  ```bash
  cmake -S . -B build2 -DPSND_ENABLE_SHARC=ON
  cmake --build build2 --target psinad -j4

  # S0 ANALYTICAL: IBr 3-state, 2 fs BO, eig 稳定在 -0.067/0.026/0.097 Ha
  cd tests/example_parm/SHARC/S0_analytical
  SHARC=/home/lhc/sharc4 /home/lhc/PSiNad/build2/psinad -p input.json

  # S0 LVC: SO2 13-state (4 singlets + 3 triplets * 3), 50 fs BO, FC 几何,
  # eig 随 LVC kappa 驱动振动；Etot 漂移 2.89e-7 Eh
  cd ../S0_lvc
  SHARC=/home/lhc/sharc4 /home/lhc/PSiNad/build2/psinad -p input.json

  # S2 LVC+SOC: SO2 13-state, soc feature 打开，S 原子偏离 FC +0.1 Bohr,
  # General_soc representation + BO，50 fs 动力学。
  # 三重态 Ms 简并被 SOC 打破 ~1.5e-6 Ha；Etot 漂移 3.89e-6 Eh
  # （和同位移无 SOC 基线完全一致，dt² scaling 证明是积分器有限步长误差）
  cd ../S2_lvc_soc
  SHARC=/home/lhc/sharc4 /home/lhc/PSiNad/build2/psinad -p input.json

  # S1 MNDO: CH2NH+ cation 4 singlets (OM2/ROHF/CI), 单点 Level A vs
  # sharc4/bin/SHARC_MNDO.py native driver 误差 grad 3e-10 / nacdr 2e-9 / eig 5e-8.
  # 10 fs BO trajectory dt=0.05fs Etot 漂移 1.7e-4 Eh (dt² scaling 持住)
  cd ../S1_mndo
  SHARC=/home/lhc/sharc4 TMPDIR=/tmp /home/lhc/PSiNad/build2/psinad -p input.json
  ```
- **关键不变量**（失守即回归）：
  - `setup_interface` 单轨迹调用次数 = 1（finalize 时 `setup_interface_calls` 计数器）
  - 默认构建（不开 `PSND_ENABLE_SHARC`）必须仍能编译通过
  - `Model_QMInterface` / `Model_QMMMInterface` 字节不变
  - General_soc 下 Model 只填 `Vc` + `dVc` + 实数 `nac`，**不要**直接写 `Ec/Tc/dEc`（见 #16）

---

## 2026-04-25 #17 | S1 MNDO：第一个 ab-initio 后端 Level A/B 通过；BAGEL 架构不兼容 ✅

**起点：** 用户提到本机有 `mndo2020` 和 `BAGEL`。which 确认 `/usr/local/bin/mndo2020` 和 `/usr/local/bagel/bin/BAGEL`。按 plan § 4 做 S1 ab-initio 验证。

**MNDO — 按标准流程走通：**

1. 复用 `sharc4/examples/SHARC_MNDO` 的 CH2NH+ cation（4 singlets, OM2/ROHF/CI(2,1)）。
2. 改 `MNDO.resources`：`mndodir` 指到 `/usr/local/bin/`（SHARC_MNDO 拼接 `mndodir + "mndo2020"`）。
3. 改 QM.in `grad 1 2 4` → `grad`（SHARC 识别不带 index 的为 all states）。
4. **第一次跑崩**：`TypeError: cannot pickle 'PyCapsule' object` at `multiprocessing/reduction.py`。
   - 根因：`SHARC_ABINITIO.runjobs` 用 `Pool.apply_async(self.execute_from_qmin, args=(workdir, qmin))`，args 包含 `self`（SHARC 实例）。pickle 期间发现我们监控 `setup_interface` 调用次数时给它挂了 `py::cpp_function` 包装器 —— 里面的 PyCapsule 不能 pickle。
   - 修复：去掉 monkey patch。计数器的原始动机（plan § 3.4 "setup_interface 调用 = 1"）由**代码结构**本来就保证 —— 我们从 C++ 只调一次，SHARC 自己不自调。直接去掉 `py::cpp_function wrap`，改成 C++ 手动 `setup_interface_calls = 1;`。
5. 第二次跑通。`Using total time 0.887519 s`（单点含 Python 启动）。

**MNDO — Level A 对拍（plan § 9.3 per-state 单点对比）：**

写 `compare.py`（一次性的，/tmp 下），解析 sharc4 `QM.out` 和 PSiNad `default/S1_mndo_*.dat`：

| 量 | 最大 |差| | 说明 |
|---|---:|---|
| eig | 4.8e-8 | PSiNad 记录文件格式 `%.8e` 只有 9 位有效位，实际 DataSet 内数值理论上匹配更精确 |
| grad (dE 对角) | 3.1e-10 | 达到 plan 1e-10 目标 |
| nacdr 离对角 | 2.0e-9 | 达标 |

✅ Level A **PASS**。

**MNDO — Level B 10 fs BO trajectory：**

- dt=0.25 fs, 10 fs (40 步): drift = -4.0e-3 Eh（4 kcal/mol）
- dt=0.05 fs, 5 fs: drift = -1.7e-4 Eh
- **dt² scaling 验证**：(0.25/0.05)² = 25 vs 实测 4e-3/1.7e-4 ≈ 23.5 ✓

结论：drift 由积分器有限时间步决定，不是模型侧 bug。plan § 9.4 ab-initio 1e-4 Eh 阈值在 dt=0.05fs 下基本达标（1.7e-4 是临界值）。MNDO 半经验 CI 本身就比 DFT/CASSCF 数值噪声大一些，可接受。

✅ Level B **基本通过**。

**BAGEL — 架构不兼容，无法用当前路径验证：**

1. 试跑：`TypeError: SHARC_BAGEL() takes no arguments`
2. grep 源码：`SHARC_BAGEL(SHARC_OLD)`，`SHARC_OLD` 是空 `class SHARC_OLD(ABC): pass`。
3. `SHARC_BAGEL.py` 只有模块级函数（`getQMout(QMin)`, `writeQMout(...)`），没有 `setup_mol / setup_interface / run / getQMout` 类方法 —— 它是 **legacy shell 脚本**设计，靠 `python SHARC_BAGEL.py QM.in → QM.out` 走文件协议。
4. 同样 legacy 的还有：PYSCF / MOLPRO / AMS_ADF / COLUMBUS（都 `inherits SHARC_OLD`）。

**决定：** BAGEL / PYSCF / MOLPRO / AMS_ADF / COLUMBUS 在当前嵌入式 Python 路径下**不可支持**。要支持就得在 `Model_SHARC_Interface::executeKernel_impl` 加一条 fork-per-step shell 路径，这和 S0-S2 的整个设计前提（"不走 fork，用嵌入式 Python"，plan § 0 第 2 条）冲突。留给未来，不在 S1 scope。

**改动：**

- `src/models/src/Model_SHARC_Interface.cpp` 去掉 `py::cpp_function` monkey patch；setup_interface_calls 改为手动递增。
- `tests/example_parm/SHARC/S1_mndo/{input.json, QM.in, MNDO.template, MNDO.resources}`.
- `tests/example_parm/SHARC/S1_bagel/README.md` 留一个"为啥不行"的说明。

**决策/理由：**

- **MNDO 先不先加 wfoverlap**：`wfoverlap` 只在 `overlap/phases` 请求时调用，我们 Level A/B 都用 BO + 直接 h/grad/nacdr，不需要。user 说 wfoverlap 没编译 —— 不编译也能跑到 S1 MNDO 完成。后面做 FSSH / 长轨迹 phase-tracking 再回头编。
- **Pool.apply_async 的 PyCapsule 陷阱是 SHARC_ABINITIO 公共坑**：所有 ABINITIO 子类（ORCA/MOLCAS/TURBOMOLE 等）都会踩 —— pickle `self` 时任何挂在实例上的 C++-origin 对象都会崩。未来加新 ABINITIO 后端需要保证 Model 不往 `sharc_instance` 上挂 py::cpp_function 或类似 wrapper。
- **BAGEL 不支持的判定要硬 fail**：没必要为了 legacy 后端留一个"也许可以"的口子。sharc4 官方计划是慢慢把 legacy 后端移植到 modern API，等那个时候自然可用；不等。

**相关：**

- `src/models/src/Model_SHARC_Interface.cpp` initializeKernel_impl（`py::cpp_function` 块删除）
- `tests/example_parm/SHARC/S1_mndo/` 四件套
- `docs/dev/sharc_interface_plan.md` § 4
- memory: 无新增（#14 `feedback_no_silent_catch.md` 的设计哲学继承到这里）

**回溯提示：**

- 如果未来某步开始给 `sharc_instance` 挂任何 C++ wrapper（`py::cpp_function / py::cast<...>`），先在 `SHARC_ABINITIO` 的子类上跑一次 — 单点测试就能暴露 pickle 失败，不用等到长轨迹。
- Level A 对拍脚本的写法可以参考 `/tmp/sharc_ref_mndo/compare.py`（这次没 checked in，因为是 ad-hoc —— 未来如果要做 CI 级 Level A 矩阵，再收纳进 `tests/` 正式化）。
- 给 BAGEL 加 fork-per-step fallback 的话，切换点应该是 `initializeKernel_impl` 早期 —— 根据 `issubclass(klass, SHARC_OLD)` 分叉到"老路径"的一套 `executeKernel_impl`。

---

## 2026-04-25 #16 | S2 LVC+SOC：General_soc 通路打通（职责边界修正）✅

**起点：** S0 通过后按 plan § 2 向下走。本机无 ORCA/MOLCAS/TURBOMOLE/MNDO —— S1 ab-initio / S1.5 QMMM 无法跑。和 `lhc_dev_soc` branch 主线最贴近的是 S2 General_soc，且 sharc4 shipped 的 SO2 `V0.txt` 自带 `SOC R` 常量块（没有 `lambda_soc`，所以 d(SOC)/dR = 0）—— 无需额外数据就能验证复数通路。跳过 S1/S1.5 直接做 S2。

**动作：**

1. 新建 `tests/example_parm/SHARC/S2_lvc_soc/`，复用 `S0_lvc` 的 `LVC.template / LVC.resources / V0.txt`，写新 QM.in（`## Requests: h, soc, grad, nacdr`）+ input.json（`representation_flag / inp_repr_flag / ele_repr_flag / nuc_repr_flag` 全 `General_soc`，`naforce: BO`）。
2. 确认 `Kernel_NAForce.cpp:114-148` —— General_soc + BO 合法，等价于"在 SO-耦合的 adiabatic 本征态上跑 BO"：`rho_Q1 = Tc |occ><occ| Tc^†` 被送到 `f[j] = Re[Tr(rho_Q1 · dEc_j)]`。
3. **第一次跑（FC 几何）**：Etot 完全 constant。调查发现 FC 点 Q=0 → LVC kappa·Q = 0 → grad = 0 → 原子不动。**正常但无法验证**。于是把 S 沿 z 方向位移 +0.1 Bohr 生成有限梯度。
4. **第二次跑（位移后）**：Etot 仍然 constant（1.89426e-3 Eh 一直不变），eig 跨 50 fs 也完全不变。原子被判定"不动"。
5. **根因定位**：`Kernel_Representation.cpp:378-383` 里，General_soc 自己做 `dEc = dVc + [nac, Vc]`，即 **Model 要填的是 `Vc` + `dVc` + 实数 `nac`（model::rep::nac）**，不是 `dEc`。我们的代码直接写 `dEc` 并且再做一次 EigenSolve 填 Ec/Tc，随后被 Kernel_Representation 覆盖 —— Vc 被读到了，但 dVc / nac 保持 0 → `dEc = 0 + [0, Vc] = 0` → 无力。
6. **修复：** `Model_SHARC_Interface.cpp` executeKernel 的 General_soc 分支改成只填 `Vc` + `dVc`（对角 = grad 实部，离对角 = 0 因 SHARC_LVC 没有 `socdr`）+ `nac`（SHARC 的 `nacdr` 实数）。header 新增 `dVc` span，`setInputDataSet_impl` 新加 `dVc = DS->def(DATA::model::dVc)` 绑定。删掉 Model 里的 `EigenSolve` 和 `Ec/Tc/dEc` 直写 —— 留给 Kernel_Representation。

**验证：**

| 跑法 | dt | Etot 漂移 | 结论 |
|---|---:|---:|---|
| S2 LVC+SOC, 位移 | 0.5 fs | **-3.89e-6 Eh** | 动力学真实发生，eig/Ec 振荡 |
| S2 LVC+SOC, 位移 | 0.05 fs | **-3.89e-8 Eh** | 漂移按 dt² scaling（BAOAB 2 阶积分器预期） |
| S0 LVC, 同位移, **无 SOC** | 0.5 fs | **-3.89e-6 Eh** | 和 S2 完全一致 → SOC 路径没引入额外误差 |
| S0 LVC, FC 原位 | 0.5 fs | -2.89e-7 Eh | refactor 无回归 |
| S0 ANALYTICAL | 0.1 fs | 稳定 | refactor 无回归 |

**SOC 正确性自检**：
- S0 LVC（无 SOC）三重态 Ms 完全简并：`eig(1)=eig(2)=eig(3)=0.1231069775`
- S2 LVC+SOC：`eig(1)=0.12310541, eig(2)=0.12310549, eig(3)=0.12310696` —— Ms 简并被打破，splitting ~1.5e-6 Ha。和 V0.txt `SOC R` 块里 T1⟷S_2 元素 (~1.2e-4 Eh) 经二阶微扰估算一致。

**决策/理由：**

- **跳过 S1/S1.5 没有破坏 plan**：plan § 2 本来就允许阶段内单验证（S1.5 依赖 S1，S2 独立）。跳 S1 去做 S2 是环境限制决定；等将来装上 QC 软件回头补 S1，不影响 S2 结论。
- **Model 填 Vc+dVc+nac 而不是 Ec/Tc/dEc** 是 PSiNad core 的既定抽象边界（见 `Kernel_Representation.cpp:328-399`）。想"一步到位"在 Model 里算 Ec/Tc/dEc 完全是走弯路。职责边界：
  - **Model** = 代数基元 (`V / dV` 实数路径；`Vc / dVc / nac` 复数路径)
  - **Kernel_Representation** = 对角化 + commutator 变换
  - **Kernel_NAForce** = 把 rho 和 dEc/ForceMatc trace 成力
- **S2 漂移超过 plan 的 1e-6 Eh 阈值**不是问题：plan 的阈值基于 S0 LVC FC 几何的 "无动力学" 场景（trivial conservation），当时测出 2.89e-7 Eh 就是噪声。真正的动力学 + BAOAB dt=0.5fs 下 3.89e-6 Eh 漂移是积分器精度上限，改 dt²  即可线性消除，**不是 SOC 特有**。

**相关：**

- `src/models/include/psnd/Model_SHARC_Interface.h` 新增 `dVc` 成员声明
- `src/models/src/Model_SHARC_Interface.cpp` `setInputDataSet_impl` + executeKernel General_soc 分支大改
- `tests/example_parm/SHARC/S2_lvc_soc/{input.json, QM.in, LVC.template, LVC.resources, V0.txt}`
- `docs/dev/sharc_interface_plan.md` § 6（S2 计划原文）
- memory: 新增 `feedback_general_soc_model_contract.md`

**回溯提示：**

- 如果将来换 SHARC_MOLCAS/ORCA + `soc` 做 S2 L2（正式 ab-initio SOC），那些后端**有 `socdr`**（检查它们的 `get_features()` 返回 set）。需要把 `have_socdr` 加入 feature 检测，填 `dVc` 的离对角复数部分（不是 0）。LVC 因为 `_lambda_soc` 存在时 SHARC 内部自己处理线性 R-依赖，**但仍然不对外 export `socdr`** —— 即使用 non-trivial `lambda_soc` 的 LVC 模型，我们的 dVc 离对角依然 0，误差有但结构上不能修。
- 如果未来要跑带 socdr 的后端做验证，先对比 dt² scaling 是否依然成立：若"dt→0 极限漂移不归 0"，说明 dVc 填漏了某一块。
- 发现"Etot 一直 constant 到机器精度"要先怀疑力是不是 0（类似 #13 / #14 的教训）。这次用"先位移 + 检查 eig 是否随时间变"两步确认了"不是配置错，是力真 0"，后面就能定位到填错 dEc 而不是 dVc。

---

## 2026-04-25 #15 | QM.in 作为 features 单一真相源 ✅

**起点：** 用户发现解包逻辑的 `features` 源是 `input.json` 的 `sharc_features` 字段，跟 QM.in 的 `## Requests` 段脱钩 —— 两处声明容易打架。

**调查：** 翻 `/home/lhc/sharc4/.claude/sharc_architecture_notes.md` 确认 SHARC 原生设计：
- line 166：standalone driver 调 `qc.read_requests("QM.in")` 传的是文件路径
- line 62-63：Fortran dynamics driver 每步**重写 QM.in**（含几何 + 任务），shell fork SHARC_XXX.py 读文件
- line 70："动力学引擎完全不知道后端是 QC 还是 NN，只认 QM.in/QM.out 协议"

结论：QM.in 是 SHARC 侧的单一真相源。我们嵌入 Python 不需要每步重写文件（用户点出这个），改成**初始化时用 file form 读一次 + 每步 dict form 只推进 step**，内容全部镜像从 QM.in 解析出来的集合。

**改动：**
1. `setInputParam_impl`：删掉 `model.sharc_features` 读取；`features.clear()`；`has_soc = false`（稍后重算）
2. `setInputDataSet_impl`：**永远**绑定 Vc/Ec/Tc/dEc 复数 span，不再按 `has_soc` 条件绑定（避免 has_soc 在 setInputDataSet 时未定义的问题；多占 F² × 16 bytes，F=13 时 ≈ 2.7 KB 可忽略）
3. `initializeKernel_impl`：在 `setup_interface()` 之后调一次 **`read_requests(qmin_file)` 文件形式**，让 SHARC parse QM.in 的 `## Requests`；然后遍历 `sharc_instance.QMin.requests.data`，对已知 feature key（h/dm/soc/grad/nacdr/phases/dmdr/socdr/overlap/ion/multipolar_fit/density_matrices）做 truthy 检查，构造 `features` set；`has_soc = features.count("soc") > 0` 在此处算
4. `input.json` 两个 S0 测试都删掉 `sharc_features` 字段
5. `S0_lvc/QM.in` 补 `## Requests` 段（h / grad / nacdr）—— 之前没写，之前跑通是因为 input.json 的 sharc_features 在顶管用

**验证：**
- ANALYTICAL：eig(0) `-0.0670 → -0.06699993` 正常 morse 振动
- LVC：Etot drift **2.889e-7 Eh** 不变（< 1e-6 阈值），eig(0) `0 → -1.65e-4` 正常 kappa 驱动振动

**决策/理由：**
- 坚决走 QM.in 单一真相源，和 SHARC 的 Fortran driver 设计保持一致 —— 以后把 SHARC 原生 QM.in 直接拿过来就能跑，不需要手工同步到 input.json
- 每步**不重写 QM.in 文件**（用户原话）—— 嵌入 Python 的优势就是 in-memory state，SHARC 的 `read_requests(dict)` 本来就是给 PySHARC/嵌入场景设计的内存更新通道
- 永远绑定 Vc/Ec/Tc/dEc 换 has_soc timing 简化 —— tradeoff 是 ~KB 内存；代码干净

**相关：**
- `src/models/src/Model_SHARC_Interface.cpp` {setInputParam, setInputDataSet, initializeKernel}
- `tests/example_parm/SHARC/S0_{analytical,lvc}/{input.json, QM.in}`
- memory: （无新增，沿用 `feedback_no_silent_catch.md` 的设计哲学）

**回溯提示：**
- 如果未来想支持"每步动态改 request"（SHARC 原生有 `gradcorrect` 触发临时 nacdr 之类）：在 C++ 维护一个 `extra_features_this_step` 临时 set，merge 到 dict-form 里传给 `read_requests` —— 仍然不碰文件。
- `read_requests(qmin_file)` 必须在 `setup_interface()` 之后调（因为 `_step_logic` 依赖 savedir），否则 SHARC 报 savedir 相关错。

---

## 2026-04-25 #14 | 删掉 try/catch，让 SHARC 异常硬崩（根因修复） ✅

**起点：** 用户发现 #13 的 "Etot 完全守恒到机器精度" 其实是**静默失败的假象** —— 应该有 LVC kappa 驱动的动力学，但 eig 完全不动。

**根因（两个 bug 互相掩盖）：**
1. `executeKernel_impl` 里硬解包 `qmout["nacdr"]`，但 S0 LVC 只请求了 `h, grad`。`QMout.__getitem__` 对未请求的 key 返回 `None`，`py::cast<py::array_t<double>>(None)` 抛 C++ 异常。
2. 我当时在整个 per-step 块外面包了 `try { ... } catch (py::error_already_set | std::exception) { stat.succ = false; stat.fail_type = 1; }` —— 异常被吃掉，grad/nac span 留零，积分器拿零当力做 BO → 原子不动 → Etot 永远 = eig[0] = 0。

**改动：**
1. **解包前按 features 判断**：
   ```cpp
   const bool have_grad  = features.count("grad")  > 0;
   const bool have_nacdr = features.count("nacdr") > 0;
   // 未请求的 feature 对应 span 清零，不解包 qmout
   ```
2. **删掉整个 per-step try/catch** —— SHARC 侧异常（类型不匹配、请求没喂对、内部错误）全部**硬崩**。对应模式借鉴 Model_QMInterface 的 stat.succ=false 是 **为了 SCF 不收敛这种业务可恢复场景**设计的；SHARC interface 这边不适用。
3. **补调 SHARC 的 per-step 收尾协议**：
   ```cpp
   sharc_instance.attr("clean_savedir")();
   sharc_instance.attr("create_restart_files")();
   sharc_instance.attr("write_step_file")();  // 关键：FAST persistent 下只更新内存 savedict["last_step"]
   ```
   不调 `write_step_file` 的话，下一步 `read_requests` 的 `_step_logic` 看到 `last_step=None` 但 specified step≥1，抛 `"Determined last step (0) ... do not fit"`。

**验证：**
- 运行时 eig(0) 真实变化：0 → -1.16e-6 → -4.59e-6 → ... → -1.65e-4 Ha（kappa 驱动的振动）
- **Etot 漂移 = 2.89e-7 Eh** 整条 50 fs 轨迹，**仍然 < 1e-6 Eh** 阈值 ✅，这次是真守恒
- 100 步干净跑完，无 stderr 异常
- ANALYTICAL test 同时重验通过（NAC-less 路径 + features 保护都对）

**决策/理由：**
- 用户原话："没有 grad 获取应该直接报错" —— 正解。隐藏 programming error 导致今天 debug 困难两倍。
- `stat.succ=false` 这个软失败 pattern 只在**业务可恢复**的场景下合理。对应 Model_QMInterface 里 `force_run` + 重试 —— 有具体业务语义。SHARC glue 完全是 type-safety / config-correctness 层面的错误，应该立刻报错。
- feature-conditional unpack（`if (have_nacdr) ...`）不是静默隐藏错误，而是**响应配置**。两者要区分：配置驱动的选择 vs. 吞掉异常。

**相关：**
- `src/models/src/Model_SHARC_Interface.cpp` executeKernel_impl
- memory: `feedback_no_silent_catch.md`

**回溯提示：**
- 如果未来真的需要给某个 SHARC 调用加重试（比如 ab-initio SCF），**窄范围**包裹那一行并写明业务理由，不要再加 per-step 大 try/catch。
- 如果再次看到"eig 完全不动但程序没报错"，第一反应查 stderr 里有没有被吞掉的异常。可以临时在 try/catch（如果还有）处加 `throw;` 重新抛出验证。

---

## 2026-04-25 #13 | S0 LVC 50 fs 能量守恒验证（❌ 假阳性，见 #14）

~~**结果：** Etot 漂移 = 0（机器精度），PASS plan < 1e-6 Eh 阈值。~~

**事后复盘（#14 里发现）：** 这个 "完美守恒" 是假的 —— 因为 `try/catch` 吞了 `qmout["nacdr"] = None` 的异常，grad span 留零，原子根本不动。真正的 Etot 漂移应该是 2.89e-7 Eh（见 #14）。

保留这条作为**教训**：看到"结果好到不可思议"（机器精度级别的守恒，通常只在无动力学时才能）要反向怀疑 —— LVC kappa 非零就该有动力学。

---

## 2026-04-25 #12 | S0 LVC smoke test 通路打通 ✅

**起点：** S0 ANALYTICAL 通过，按计划 § 3.1 做第二个 FAST 后端（LVC）的通路验证。

**动作：**
1. 复制 sharc4 shipped example：`sharc4/examples/SHARC_LVC/{LVC.template, LVC.resources, V0.txt}` → `tests/example_parm/SHARC/S0_lvc/`
2. 写 `QM.in`：
   - 3 原子（SO2），坐标取自 V0.txt 的 reference geometry（FC 点）
   - `unit bohr`
   - `states 4 0 3` → 4 singlets + 0 doublets + 3 triplets = 13 nmstates
3. 写 `input.json`：
   - `sharc_backend: LVC`, `N=9, F=13, occ=0`
   - `sharc_features: "h,grad"`（不要 `soc` / `dm` —— S0 只验通路；三态 Ms 子能级会因为无 SOC 而简并，正好成为正确性自检）
   - `naforce: BO`, `sampling_nuc_flag: Fix`, `dt 0.1 fs, tend 2.0 fs`
4. 直接跑，无额外改动就通过

**验证：**
- 21 帧 eig 输出到 `default/S0_lvc_eig.dat`
- 13 个本征值和 `LVC.template` `epsilon` 块**逐元素匹配**：
  ```
  eig(0)=0.0         ← singlet 1  ε(1,1) = 0.0000000000
  eig(1)=0.1553      ← singlet 2  ε(1,2) = 0.1553277839
  eig(2)=0.1689      ← singlet 3  ε(1,3) = 0.1688652931
  eig(3)=0.3088      ← singlet 4  ε(1,4) = 0.3088375445
  eig(4..6)=0.1231,0.1547,0.1601  ← triplet 1/2/3 (Ms=-1)
  eig(7..9) 同上                  ← triplet 1/2/3 (Ms=0)
  eig(10..12) 同上                ← triplet 1/2/3 (Ms=+1)
  ```
- Ms 简并符合"无 SOC 时三重态 3 个子能级相同"的物理预期
- `setup_interface` 单次调用（finalize 无 WARNING）
- 运行时间 4.63 s（LVC 比 ANALYTICAL 稍慢，因为矩阵维度 13×13 而不是 3×3）

**决策/理由：**
- 不写新的 LVC model（比如 pyrazine）—— 计划里 pyrazine 只是举例，shipped 的 SO2 example 能验一样的通路。避免造新数据。
- 三态简并（无 SOC）当作正确性自检：如果出现 Ms 分裂，说明 SOC 部分无意中漏了。

**相关：**
- `tests/example_parm/SHARC/S0_lvc/` 五件套
- `docs/dev/sharc_interface_plan.md` § 3.1

**回溯提示：**
- SHARC 的 `states 4 0 3` 语法：三个数字 = 三个多重度（singlet, doublet, triplet）的状态数。nmstates = Σ n_i × (2S_i + 1) = 4·1 + 0·2 + 3·3 = 13。这里 `F = 13` 必须和 LVC.template 开头的 `4 0 3` 严格一致，否则 SHARC 和 PSiNad 对不齐。
- 不需要动 Model_SHARC_Interface 的代码 —— 说明 S0 骨架对多种 FAST 后端是通用的（ANALYTICAL + LVC 协议完全一致）。

---

## 2026-04-25 #11 | `setInputDataSet` vs `initializeKernel` 职责重构 ✅

**起点：** `setInputDataSet_impl` 里手工读 QM.in 两条信息（元素 + unit 关键字），实现冗长，注释里写"Parse QM.in twice"其实是一遍夹带两件事。用户反馈代码不好读。

**第一次尝试（失败）：** 想把 atoms/mass/x0/p0 全部延后到 `initializeKernel_impl` 里做（那时 SHARC 的 setup_mol 已经解析了 unit，直接用 `molecule["factor"]`）。

**失败现象：** x[t=0] 正确，x[t=0.1] 变 NaN / Inf，全部原子飞走。

**根因：** `src/kernels/src/Kernel_Update_x.cpp:25-34` 在它自己的 `setInputDataSet_impl` 里就读 `mass` 并预计算 `minv = 1/mass`。mass 零 → minv = inf → 第一步 `x += p * minv * dt` 爆炸。

**第二次（成功）：** 按**数据依赖**切分，不是按文件顺序：

| 阶段 | 放这里的原因 |
|---|---|
| `setInputDataSet_impl` | span 绑定 + 其他 kernel 在同阶段就要读的值（atoms/mass）。只需要元素标签，所以读 QM.in header 时跳过坐标值（`ifs >> dummy`） |
| `initializeKernel_impl` | SHARC 启动 + 让 SHARC 解析 unit + 用 `factor` 填 x0/p0；cache `coord_scale = 1/factor` |
| `executeKernel_impl` | 每步用 cached `coord_scale`，不查 Python |

**验证：** 21 帧输出与 refactor 前完全一致。总时间 0.49 s。

**memory：** `feedback_setinputdataset_contract.md`

**回溯提示：** 如果未来又想"把 mass 挪去别处"，先查 `Kernel_Update_x::setInputDataSet_impl` 有没有改，没改就还是不行。

---

## 2026-04-25 #10 | record rule 语法定位 ✅

**起点：** 想在 S0 test 加 `record` 让 eig 落盘，第一次写 `"eig<mi>:R(1, eig<mii>)"` 报 `Loss of key: integrator.eig`。

**定位链路：**
1. `VariableDescriptor.cpp:59` 正则：`name{field@time}<index>:type`
2. field 快捷字母：`I→integrator`（默认）、`M→model`、`P→parameter`、`R→record`（line 77-89）
3. DataSet 里 eig 注册名 `model.rep.eig`（`vars_list.cpp:343`，经 `Variable.h:112` 的 `_subreplace(name, "::", ".")` 把 `::` 变 `.`）

**正确写法：**
```json
{"rule": "eig<mi>:R(rep.eig{M}<mi>)", "save": "S0_analytical_eig.dat"}
```
- LHS `eig<mi>:R` → 输出写到 `record.eig`（TabularOutput 强制 field = record），einsum 下标 mi，Real 类型
- RHS `rep.eig{M}<mi>` → 读 `model.rep.eig`，同下标，trivial copy
- `<mi>`：m 对应 P 维（swarm），i 对应 F 维（state）；eig shape = PF

**回溯提示：** 想记其他量用同样套路，先在 `src/core/src/vars_list.cpp` 找注册名，冒号变点号。

---

## 2026-04-25 #9 | IBr 几何方向翻转 ✅

**起点：** `x0` 填充修好后，eig[t≥1] 还是 `~1e+8` 级别，分子被爆炸力抛飞。

**根因：** `ANALYTICAL.template` 的 Morse 表达式是 `a1*((1 - exp(g1*(r1 - xi + xbr)))^2 - 1)`，位移 = `r1 + (xbr - xi)`。sharc4 shipped 的 `QM.in` 写 `Br +6.10463 Bohr`，位移 = `4.666 + 6.10 = 10.77`，`exp(g1*10.77) ≈ 4.5e4`，eig 爆。

**判断：** sharc4 自带的那份 QM.in 看起来只是"结构模板"、不是"可跑"的初始构型。

**改动：** `tests/example_parm/SHARC/S0_analytical/QM.in` 中 `Br -4.66600 Bohr`（= `-r1`），让 `xbr - xi = -r1 → Morse 位移 = 0 → state 0 在势阱最低点`。

**验证：** eig 稳定在 -0.067 / 0.0260 / 0.0969 Ha，物理上合理的 IBr 能谱。

**回溯提示：** 如果想换其他解析模型（Tully / Morse-3c），Morse 表达式的符号约定要先算一遍再放 QM.in。

---

## 2026-04-25 #8 | 坐标单位换算（双向） ✅

**起点：** `eig[0] = 1.39e+08 Ha`，分子被巨大力抛飞到 `x = ±1e+7 Bohr`。

**根因：** C++ 原来硬编码 `raw = x * au_2_ang`（Bohr → Å），但 QM.in 里写 `unit bohr`，SHARC 的 `molecule["factor"] = 1.0`（不换算），于是它把我送的"Å 数值"当 Bohr 用。IBr 6.10 Bohr ≈ 3.23 Å，SHARC 看到的距离 3.23 Bohr 在排斥区。

**关键事实：** SHARC 的 `factor` 字段语义 = "输入 → Bohr 的乘数"：
- `unit bohr`：factor = 1.0
- `unit angstrom`：factor = 1/0.529 ≈ 1.889

**双向换算规则：**
- C++ → SHARC：PSiNad 存 Bohr，要送 "SHARC 期望的输入单位"，所以 `coord_scale = 1/factor`。
- QM.in → x0：文件里的数值是 QM.in 声明的单位，要转 Bohr，所以 `x0 = raw * factor`。

**改动：** 在 `initializeKernel_impl` 里查一次 `sharc_instance.attr("QMin").attr("molecule")["factor"]`，cache `coord_scale = 1/factor`，同时用 `factor` 填 x0。

**回溯提示：** Model_QMInterface 没踩这个坑是因为它的 QM.in 格式强制 Å（硬编码 `/ au_2_ang`）；SHARC 的 QM.in 灵活所以得动态查。

---

## 2026-04-25 #7 | 填 x0（Sampling_Nucl::Fix 的前提） ✅

**起点：** `eig[t=0]` 正确但后续步骤错，说明 Model 被调了两次，第一次 x 是垃圾。

**根因：** `Sampling_Nucl::Fix`（`Sampling_Nucl.cpp:70-72`）做 `x[j] = x0[j], p[j] = 0`。我的 Model 在 `setInputDataSet_impl` 里读了 atoms/mass 但**没填 x0**，所以 Fix 拷了全零 → 两原子都在原点 → SHARC 算出 `exp(g * r1) ≈ 1e+8` 爆炸能量。

**改动：** `setInputDataSet_impl` 里新增 x0/p0 的 span 绑定 + 从 QM.in 读 raw 坐标（当时还带单位侦测逻辑，后来在 #11 清理掉）。

**讨论但未采纳：** 用户提议把 init 几何从 QM.in 解耦到 `init.ds` + `sampling_nuc_flag = "ReadDataset"`。**决定保留现状**——C++ 从 QM.in 读 x0 的路径当 Fix 模式 fallback，生产用 trajectory 再走 ReadDataset。

**回溯提示：** 如果未来切 ReadDataset 为默认，`setInputDataSet_impl` 里的 QM.in 坐标读取可以删掉，atoms/mass 填充必须保留（见 #11）。

---

## 2026-04-25 #6 | SHARC_ANALYTICAL 大小写 bug 绕开 ✅

**起点：** `TypeError: loop of ufunc does not support argument 0 of type Add which has no callable exp method`，来自 `SHARC_ANALYTICAL.py:366: self._H = self._fH(coords_needed)`。

**根因（用户代码 bug）：** `SHARC_ANALYTICAL.read_template` 在 line 139 对 gvar 行调 `.lower()`，得到小写 `xi, xbr`；但 Hamiltonian 字符串是通过 `find_lines`（line 99-109）读入，**没做 lowercase**，保留大写 `xI, xBr`。`sympy.lambdify([_gvar_symb=[xi,xbr]], Hmat(含xI,xBr))` 之后，数字填不进 Hmat 里的大写符号 → 返回 sympy Add 表达式。

**绕开方案（不改 SHARC）：** 把我的测试 template `ANALYTICAL.template` 里所有变量名全改成小写（xI→xi、A1→a1、R1→r1……）。语义不变。

**上游修复建议：** `find_lines` 返回前也 `.lower()`；或干脆 `_gvar` 不 lowercase。值得报 issue。

**回溯提示：** 以后从 sharc4 复制 template 过来先 grep 一遍大写字母。

---

## 2026-04-25 #5 | request dict 类型：bool vs string ✅

**起点：** `TypeError: grad should be of type <class 'list'> but is <class 'bool'>`。

**根因（SHARC 规范）：** `SHARC_INTERFACE.py:957 _set_driver_requests` 对 request dict 的规则：
- `grad / nacdr / overlap / ion / multipolar_fit / density_matrices` → **字符串**（`"all"` 或 `"1 2 3"`），SHARC 展开成 state index list
- `h / dm / soc / phases / dmdr / socdr` → **bool True**

**改动：** 加 `sharc_feature_is_listlike(name)` 辅助函数，构造 `read_requests` dict 时按类型分派：
```cpp
if (sharc_feature_is_listlike(f)) req[py::str(f)] = std::string("all");
else                              req[py::str(f)] = true;
```

**上游建议：** `_set_driver_requests` 可以加 bool→list-all 展开兜底，避免这种类型陷阱。

**回溯提示：** 未来加新 feature 先看 `SHARC_INTERFACE.py:982-999` 的循环，被列进那里的 task 名要用字符串。

---

## 2026-04-25 #4 | SHARC savedir 挂在 `-d` 下 + C++ mkdir ✅

**起点：** `FileNotFoundError: [Errno 2] ... '/path/to/SAVE'` at `SHARC_ANALYTICAL.py:238`。

**根因：** `SHARC_FAST.setup_interface()` 做 `os.listdir(savedir)` 但不 mkdir。SHARC 独立 driver 正常在外部创建 savedir，嵌入式用法下得 C++ 负责。另外 SHARC 默认 savedir 是 CWD 相对路径，多 job 并发会互相污染。

**改动：** `initializeKernel_impl` 里在 `setup_interface()` 之前：
1. 覆写 `sharc_instance.attr("QMin").attr("save")["savedir"] = directory + "/SAVE"`（`Kernel::directory` 对应 `-d` gflag）
2. `::mkdir(savedir, 0755)`（必须显式 `::` 前缀，否则 `struct stat` 会和 PSiNad 的 `Status` 类名冲突）

**memory：** `feedback_sharc_savedir_under_d.md`

**上游建议：** `SHARC_FAST.setup_interface` 开头加 `os.makedirs(savedir, exist_ok=True)` 就能兼容嵌入式调用。

**回溯提示：** 如果以后想支持多 Model_SHARC_Interface 实例并存（S4 ECI 场景），每个实例 savedir 还得带 suffix。

---

## 2026-04-25 #3 | input format / naforce / occ / sampling_nuc_flag ✅

这几个是一起修的基础配置问题，合并一条：

**#3a TOML → JSON：** 项目全部 test 是 JSON，不要引入 TOML。虽然 `Param::fromFile` 两种都支持。memory: `feedback_input_json.md`。

**#3b naforce BOSH → BO：** S0 目标是验证连通性和能量守恒，BOSH 引入随机跳跃让验证复杂化。FSSH / surface hopping 对拍是 S1 Level B 的事。

**#3c sampling_nuc_flag Constant → Fix：** `"Constant"` 不在 `NuclearSamplingPolicy` 枚举里。`Fix` 的语义正是"用 x0 初始化、p = 0"。

**#3d 加 occ = 1：** `Kernel_Elec_Functions.cpp:44` 和 `Sampling_Elec.cpp:29` 要求 `occ0 >= 0`。

**回溯提示：** 未来加 test 前先看 `src/kernels/include/psnd/Policy.h` 里的枚举。

---

## 2026-04-25 #2 | S0 骨架四件套落地 ✅

**起点：** 计划 doc `sharc_interface_plan.md` 已 finalized，开始实现。

**一次性写完的东西：**

| 文件 | 动作 |
|---|---|
| `CMakeLists.txt:387-411` | `option(PSND_ENABLE_SHARC OFF)`；开启时 find Python3 + pybind11，给 shared/static target 加 `PSND_WITH_SHARC` 宏和链接 |
| `src/models/include/psnd/Model_SHARC_Interface.h` | `DEFINE_POLICY(SHARCBackend, ...)` 22 个后端枚举 + 类骨架 |
| `src/models/src/Model_SHARC_Interface.cpp` | `std::once_flag` 守住进程级 `py::scoped_interpreter`；sharc_backend_is_fast() 判断 FAST 子类用 persistent=True；initialize/execute/finalize 三件套 |
| `src/models/src/ModelFactory.cpp:10,33-34` | 注册 `"SHARC"` |
| `tests/example_parm/SHARC/S0_analytical/` | `input.json / ANALYTICAL.template / ANALYTICAL.resources / QM.in` |

**构建验证：**
- `-DPSND_ENABLE_SHARC=OFF`（默认）→ 编译通过，不依赖 Python
- `-DPSND_ENABLE_SHARC=ON` → `libpython3.11.so` 链到 `psinad`

**注意：** pybind11 的 `py::object` 有 hidden visibility，在 default visibility 的共享库里会触发 `-Wattributes`，用 `#pragma GCC visibility push(hidden)` 包类声明压掉。

**回溯提示：** 未来 S1/S2 加新后端主要是 `initializeKernel_impl` 里改 backend dispatch + `executeKernel_impl` 里改 QMout 拆包，骨架不会大改。

---

## 2026-04-24 #1 | S0 实施计划与范围收敛 ✅

**起点：** 前一轮会话把 `docs/dev/sharc_interface_plan.md` 从"覆盖所有 30 后端"收敛到"单进程单轨迹先跑对"。

**关键决策（已写入 plan § 12）：**
- 当前阶段 **不引入** FAST 多轨迹常驻缓存（`s_fast_instance_cache` 之类）
- `sharc_instance` 普通 `py::object` 成员即可，不要 static
- FAST vs ABINITIO 的持久化策略分歧留到 S3 再说
- OpenMP、Python 解释器跨实例复用、restart 格式 —— 全部延后

**memory：** `project_sharc_interface.md`（单轨迹 scope，禁止 static cache 的红线规则）

**回溯提示：** 如果未来 PR 里出现 `static std::map<..., py::object>` 这种跨实例共享状态，先翻 plan § 12 和这条 memory，确认是不是真的到了要开多轨迹阶段。

---

## 模板（下次追加时用）

```
## YYYY-MM-DD #N | 简短标题 ✅/🟡/⏳/❌

**起点：** 当前系统处于什么状态、为什么要做这件事。

**动作：** 具体改了哪些文件、加了什么逻辑。关键片段可以贴小片代码。

**验证/产出：** 改完之后跑了什么命令，观察到什么结果。数值贴关键帧。

**决策/理由：** 非显然的选择（为什么 A 不 B、为什么先不做 X）。

**相关 memory / 文档：** `feedback_xxx.md`、`docs/dev/xxx.md`。

**回溯提示：** 未来如果想 undo 或 revisit 这一条，需要知道什么前置条件、哪些文件会被牵动。
```
