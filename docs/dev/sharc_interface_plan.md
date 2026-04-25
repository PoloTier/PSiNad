# SHARC_Interface 接入 PSiNad — 实施计划

> 分支：`lhc_dev_soc`（可另开 `lhc_dev_sharc`）
> 相关计划：[soc_migration_plan.md](soc_migration_plan.md)
> SHARC 架构参照：`/home/lhc/sharc4/.claude/sharc_architecture_notes.md`

---

## 0. 目标与设计原则

把 SHARC4 的 `SHARC_INTERFACE` 生态（~30 个后端：ab-initio / NN / QMMM / 组合）接入 PSiNad 动力学引擎，由 PSiNad 作为 driver，调用 SHARC 侧后端拿 `{h, grad, nacdr, soc, dm, phases}`。

**核心原则：**

1. **一个 C++ 类覆盖三类**（QM / NN / QMMM）。SHARC 侧已经把它们统一到同一抽象基类，我们不再二次抽象。
2. **嵌入式 CPython + pybind11**，不走 fork-per-step 的 shell 协议。项目已有 `thirdpart/pybind11` 和 `BUILD_PYTHONLIB` 开关，成本低。
3. **并行新增，零触碰现有路径**。`Model_QMInterface` / `Model_QMMMInterface` / `Model_Interf_MNDO` 保持字节不动，SHARC 路径由新 `Model_SHARC_Interface` 承担。
4. **构建层面可选**：`option(PSND_ENABLE_SHARC OFF)`，默认关闭；关闭时 Factory 返回 stub，不引入 Python-dev 依赖。
5. **和 `lhc_dev_soc` 正交**：SHARC 的 `'soc'` feature 给复 Hermitian H，直接流入已建好的 `Vc / Ec / Tc / dEc` 通道，不污染实数分支。按 `General_soc` 独立 block 规则组织（不能 fallthrough）。
6. **测试先行**：每阶段完成先跑协议 smoke test 再 scale up。

核心数据流：

```
PSiNad Model 生命周期              SHARC_INTERFACE 协议             数据落点
───────────────────────────────────────────────────────────────────────────────
setInputParam_impl              读配置（backend / template / ...）
setInputDataSet_impl        →   setup_mol                          bind spans
initializeKernel_impl       →   read_resources / read_template /
                                setup_interface / read_requests     构造 py::object（一次）
executeKernel_impl (每步)   →   set_coords → run → getQMout        QMout → DataSet 拆包
finalizeKernel_impl         →   create_restart_files, 释放实例
```

---

## 1. 类骨架（最终形态）

```cpp
// src/models/include/psnd/Model_SHARC_Interface.h
DEFINE_POLICY(SHARCBackend,
              // ab-initio (fork QC internally)
              ORCA, MOLCAS, MOLPRO, TURBOMOLE, BAGEL, GAUSSIAN,
              COLUMBUS, NWCHEM, PYSCF, AMS_ADF, MNDO, MOPACPI, BASICORCA,
              // SHARC_FAST (in-process, persistent)
              LVC, ANALYTICAL, SPAINN, SCHNARC,
              // composite
              QMMM, OPENMM, DROPLET, ECI,
              // auto-switching
              ADAPTIVE,
              NONE);

class Model_SHARC_Interface final : public Model {
   private:
    SHARCBackend::_type backend;
    bool                has_mm;        // backend ∈ {QMMM, OPENMM, DROPLET, ECI}
    bool                has_soc;       // 'soc' ∈ features
    bool                persistent;    // FAST 子类强制 true
    std::set<std::string> features;

    // pybind11 持有的 Python 对象（RAII）。
    // 当前阶段范围：单进程单轨迹 —— initializeKernel_impl 只会被调一次，
    // sharc_instance 作为普通成员即可保证 setup_interface 只跑一次。
    // 未来如果启用单进程多轨迹（FAST 后端 ensemble），再加进程级 static 缓存（见 § 11）。
    py::object  sharc_instance;
    py::module_ numpy_mod;

    // 共用 spans（和 Model_QMInterface 同形）
    span<psnd_real>  x, p, mass, V, dV, eig, T, dE, nac, nac_prev;
    span<psnd_int>   atoms, istep_ptr, fail_type_ptr;
    span<psnd_real>  dt_ptr, t_ptr;
    span<psnd_bint>  succ_ptr, frez_ptr, last_attempt_ptr;
    psnd_real*       osc_strength;

    // QMMM 专用，has_mm 时 bind
    span<psnd_int>   layer_type;
    span<psnd_real>  point_charges;

    // General_soc 专用，has_soc 时 bind
    span<psnd_complex> Vc, Ec, Tc, dEc;

    void    setInputParam_impl(std::shared_ptr<Param>) override;
    void    setInputDataSet_impl(std::shared_ptr<DataSet>) override;
    Status& initializeKernel_impl(Status&) override;
    Status& executeKernel_impl(Status&) override;
    Status& finalizeKernel_impl(Status&) override;
};
```

---

## 2. 阶段划分与验收标准

| 阶段 | 内容 | 代码量估计 | 验收 |
|---|---|---|---|
| **S0** | 嵌入式 CPython + `SHARC_ANALYTICAL` + `SHARC_LVC` 打通 | ~300 行 C++ + 20 行 CMake | 2-state pyrazine LVC 50 fs 能量守恒 < 1e-6 Eh |
| **S1** | 加 ab-initio (`SHARC_ORCA` 为主，`MOLCAS` / `TURBOMOLE` / `MNDO` 验证不同约定) | +200 行 | 单点 QM 数值和 SHARC 原生驱动匹配到 1e-10 |
| **S1.5** | 加 QMMM（`SHARC_QMMM` with `ORCA + OPENMM`） | +150 行（`has_mm` 分支） | MeCN-水盒子 10 fs，QM+MM 梯度合并正确 |
| **S2** | General_soc 通道（`soc` feature + `Vc/Ec/Tc`） | +200 行（`has_soc` 分支） | IBr 单点 Hermitian H 对角化 → `Ec` 对齐 |
| **S3** | NN 后端（`SHARC_SPAINN`, `SHARC_SCHNARC`）、测常驻性能 | +50 行（已有通路） | 每步 < 5 ms（不含模型 inference） |
| **S4** | `SHARC_ADAPTIVE` 自动切换 + `SHARC_ECI` 复合 | +50 行 | 一条轨迹内自动 QC↔NN 切换不崩 |

每阶段完成后 commit 一次，commit message 用 `SHARC Sx:` 前缀和 SOC migration 区分。

---

## 3. S0 — 嵌入式 CPython 通路（最小骨架）

### 3.1 新增文件

```
src/models/include/psnd/Model_SHARC_Interface.h
src/models/src/Model_SHARC_Interface.cpp
tests/example_parm/SHARC/
    S0_analytical/        # SHARC_ANALYTICAL 2态1D
        input.toml
        ANALYTICAL.template
        ANALYTICAL.resources
        QM.in
    S0_lvc/               # SHARC_LVC pyrazine 4D
        input.toml
        LVC.template
        LVC.resources
        QM.in
```

### 3.2 ModelFactory 注册

```cpp
// src/models/src/ModelFactory.cpp
} else if (name == "SHARC") {
    return std::make_shared<Model_SHARC_Interface>();
}
```

### 3.3 CMake

```cmake
# src/models/CMakeLists.txt
option(PSND_ENABLE_SHARC "Embed CPython to drive SHARC interfaces" OFF)
if(PSND_ENABLE_SHARC)
    find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
    target_compile_definitions(psnd_models PRIVATE PSND_WITH_SHARC)
    target_link_libraries(psnd_models PRIVATE
        Python3::Python Python3::NumPy pybind11::embed)
endif()
```

新增 `.cpp` 后重跑 `cmake -S . -B build`（memory 里记的 glob 刷新）。

### 3.4 S0 关键实现点

**本阶段明确目标：单进程单轨迹跑起来并跑对**。多轨迹常驻是 S3 之后才考虑的事。

- **解释器生命周期**：`py::scoped_interpreter` 只在全局初始化一次（`std::once_flag` 守住），进程退出时不手动 `Py_Finalize`。
- **`initializeKernel_impl` 单次 setup**：
  - `sys.path.insert(0, $SHARC/bin)` + `$SHARC/lib`
  - `py::module_::import("SHARC_LVC")`（或 SHARC_ANALYTICAL）
  - 依次 `setup_mol / read_resources / read_template / setup_interface / read_requests`
  - 把结果 `py::object` 存成 `sharc_instance` 成员
- **`executeKernel_impl` 每步三步走**：
  - Bohr→Å，拷一份到 numpy 2D 数组
  - `sharc_instance.attr("set_coords")(coords); sharc_instance.attr("run")();`
  - `py::dict qmout = sharc_instance.attr("getQMout")();` → 拆包到 `eig / dE / nac`
- **相位跟踪**：S0 阶段沿用 `track_nac_sign`（从 `Model_QMInterface.cpp:397` 抽到 `util/nac_phase.h`）。S2 之后改用 SHARC 的 `phases` feature。
- **验证**：在 `setup_interface` 外部挂一个 Python 计数器（monkey patch 或 C++ 侧计数），**断言整条轨迹里 `setup_interface` 调用次数 = 1**。这是检测"通路是不是真的持久化"的最直接方法。
- `executeKernel_impl` 中：
  - 坐标 Bohr→Å（拷一份到 numpy，不动 `x`）
  - `inst.set_coords(coords)`
  - `inst.run()`
  - `qmout = inst.getQMout()`  —— pybind dict
  - 解包 `h` 对角 → `eig`, 非对角 → `dE`；`grad` → `dE` 对角块；`nacdr` → `nac`
- 相位跟踪：S0 阶段先沿用 `track_nac_sign`（抽到 `util/nac_phase.h` 以便多 Model 复用）；S2 之后改用 SHARC 的 `phases` feature

### 3.5 S0 验收

```bash
cd tests/example_parm/SHARC/S0_lvc
psnd input.toml
```

- 能无 crash 跑完 50 fs (dt = 0.5 fs, 100 步)
- 总能量最大漂移 < 1e-6 Eh
- 态占据曲线和 `sharc4/bin/driver.py` 跑同一 LVC 模型得到的曲线**在统计噪声内一致**（对比 expected occupation at t=10/20/50 fs）

---

## 4. S1 — ab-initio 后端

### 4.1 多后端同时加入（不是一个一个来）

理由：ab-initio 后端协议 100% 一致，只有 template/resources 不同。在 S0 打通通路之后，**加一个新后端 ≈ 加一个 enum + 一个 test 子目录**。同时加多个能暴露"约定不统一"的 bug 更早。

### 4.2 S1 必做的四个后端

| 后端 | 理由 | 测试体系 |
|---|---|---|
| **SHARC_ORCA** | 最常用 ab-initio，SHARC 侧最稳 | H2O SA3-CASSCF(6,5) 单点 + 10 fs |
| **SHARC_MOLCAS** | 激活空间约定 / 多重态约定不同 | CH2O SS-CASSCF 单点 |
| **SHARC_TURBOMOLE** | ADC(2) / TDDFT 分支 | ethene TDDFT 单点 |
| **SHARC_MNDO** | PSiNad 已有原生 `Model_Interf_MNDO` 可做对照 | 2-state 小分子，两条路径结果对拍 |

### 4.3 S1 验收矩阵

每个后端跑两个 level：

```
Level A — 单点 QM 回归：
   输入：固定几何
   对比：PSiNad-SHARC 驱动 vs sharc4/bin/driver.py 原生驱动
   判据：eig / grad / nacdr / dm 数值差 < 1e-10（浮点抖动级别）

Level B — 短轨迹：
   输入：典型初始条件（Wigner 采样的第一个构型）
   运行：50 fs surface hopping
   对比：能量守恒 (< 1e-4 Eh for ab-initio) + 最终态占据在 ±2% 以内
```

---

## 5. S1.5 — QMMM / OPENMM

### 5.1 需要回答的问题

- PSiNad 的 `x` 是扁平 `3*natom_total` 数组，QMMM 侧需要区分 QM/MM 原子。**约定沿用 `Model_QMMMInterface.h:55` 的 `layer_type` span** —— 每原子一个 int：QM=0, MM=1, link=2。
- MM 侧点电荷每步变化（MM 原子在动），`point_charges` span 必须 per-step 刷新而不是只在 `setInputDataSet_impl` 读一次。
- link 原子的 hydrogen cap 由 SHARC_QMMM 内部处理，PSiNad 不管。

### 5.2 S1.5 新增代码

- `Model_SHARC_Interface.h` 里 `layer_type` / `point_charges` 两个 span（只在 `has_mm` 时 bind）
- `executeKernel_impl` 里一个 `if (has_mm) qmin["point_charges"] = ...` 分支
- 拆包阶段 `grad` 覆盖所有原子（QM + MM），和纯 QM 分支共享解包函数，区别只在 size check

### 5.3 S1.5 测试

```
tests/example_parm/SHARC/S1p5_qmmm_orca_openmm/
    input.toml
    QMMM.template             # SHARC 原生 QMMM template
    QMMM.resources
    ORCA.template             # inner QM
    ORCA.resources
    OPENMM.template           # inner MM
    OPENMM.resources
    system.pdb
    QM.in
```

**物理体系**：水合 formaldehyde（CH2O 为 QM，水为 MM，~500 原子）。

**验收**：
- 10 fs 轨迹，QM 部分能量 + MM 动能合并后总能守恒 < 1e-3 Eh
- QM 区域原子的梯度 PSiNad-SHARC vs SHARC 原生 < 1e-8 Eh/Bohr
- MM 区域原子的梯度（来自 QM 电场 + MM-MM）< 1e-8

---

## 6. S2 — General_soc 复数通路（与 `lhc_dev_soc` 会合）

### 6.1 触发条件

```toml
[solver]
rep_flag = "General_soc"
[model]
name          = "SHARC"
sharc_backend = "ORCA"             # 或 MOLCAS
sharc_features = "h,grad,nacdr,soc,dm,phases"
```

### 6.2 数据路径（独立 block，不 fallthrough）

```cpp
if (Kernel_Representation::representation_type == RepresentationPolicy::General_soc) {
    // complex path
    fill_Vc_from_qmout(qmout);         // SHARC 返回的 'h' 已包含 SOC 虚部
    EigenSolve(Ec.data(), Tc.data(), Vc.data(), Dimension::F);
    fill_dEc_from_grad_and_nacdr(qmout);
    // 完全独立，不碰 V/dV/eig/T/dE/nac
} else {
    // real path — S0/S1 已验证
    fill_V_eig_T_dE_real(qmout);
    fill_nac_real(qmout);
    if (!stat.first_step) track_nac_sign();
}
```

### 6.3 S2 验收

**单点测试**：IBr 分子 SA-CASSCF(8,6) + RASSI SOC
- PSiNad-SHARC 驱动得到的 `Vc` 矩阵和 MOLCAS `$Project.rassi.h5` 里的 SOC 矩阵**逐元素一致**
- `EigenSolve(Ec, Tc, Vc, F)` 的本征值和 MOLCAS 报告的 spin-orbit states 能量**一致到 1e-8 Eh**

**动力学测试**：IBr 的 B/B' 曲线交叉区域 50 fs ISH（基于 SOC 的 intersystem crossing）
- 自旋多重态之间的占据转移出现且量级合理
- 对比 psinad_wbh 上 SOC1 hardcoded 模型跑的 reference curve（已有 reference data，从 Layer 3 commit 延续）

---

## 7. S3 — NN 后端 + 常驻性能验证

### 7.1 后端

- `SHARC_SPAINN`：schnetpack ≥ 2，需要预训练模型
- `SHARC_SCHNARC`：schnetpack < 1

### 7.2 性能目标

**当前阶段只做单轨迹验证**，跨轨迹复用延后（见 § 11）。

- **单轨迹常驻验证**：一条轨迹里 `setup_interface` 调用计数 **= 1**（不是每步都调）。
  实现：给 Python 实例打一个 monkey patch 计数器，断言收尾时计数 = 1。
- **每步开销 < 5 ms**（不含 NN forward）—— 证明 pybind11 调用层本身不是瓶颈
- 和 S1 的 ORCA 每步对比：NN 路径应比 ORCA 路径**快 ~100×**（典型值）

### 7.3 S3 测试

数据源：SHARC 仓库里自带的 `SPAINN_example/` 或 `SCHNARC_example/` 的 checkpoint。

```
tests/example_parm/SHARC/S3_spainn_ch2nh/
    input.toml
    SPAINN.template
    SPAINN.resources
    model_ch2nh.pt      # 预训练权重（符号链接到 sharc4 example 目录）
    QM.in
```

**验收**：100 fs 轨迹能量漂移 < 1e-3 Eh（NN 本身精度决定），轨迹曲线和 sharc4 原生驱动一致。

---

## 8. S4 — ADAPTIVE + 组合

### 8.1 SHARC_ADAPTIVE

自动在 ab-initio 和 NN 之间切换。对 PSiNad 来说是纯配置：

```toml
[model]
sharc_backend = "ADAPTIVE"
sharc_template = "ADAPTIVE.template"   # 里面指定 QC+NN 两个子 backend
```

### 8.2 S4 测试

- 一条 100 fs 轨迹，ADAPTIVE 日志里能看到 N 次 QC→NN 切换
- 能量守恒不因切换瞬间跳变
- 切换判据（不确定性阈值）从 template 读取，PSiNad 不设置

---

## 9. 测试方案（统一整理）

### 9.1 测试层次

```
Tier 1 — 构建测试（CI 最先跑）
  1.1 PSND_ENABLE_SHARC=OFF 构建通过（确认可选性）
  1.2 PSND_ENABLE_SHARC=ON 构建通过（确认 CMake 逻辑）
  1.3 PSND_ENABLE_SHARC=ON 链接时无 undefined symbol

Tier 2 — 协议 smoke test（无 QC 依赖）
  2.1 SHARC_ANALYTICAL 单点 — 嵌入式 Python 通路最小验证
  2.2 SHARC_ANALYTICAL 10 步 — 通路稳定性
  2.3 SHARC_LVC 单点 — FAST 子类 + 常驻
  2.4 SHARC_LVC 100 步 — 性能基线

Tier 3 — 后端回归（需要对应 QC 软件）
  3.1 单点数值对拍（level A）
      对每个 backend × {ORCA, MOLCAS, TURBOMOLE, MNDO, LVC, SPAINN, SCHNARC,
                       QMMM(ORCA+OPENMM), ANALYTICAL}
  3.2 短轨迹对拍（level B, 50 fs）
      对每个 backend，能量守恒 + 态占据末值

Tier 4 — 物理集成测试
  4.1 General_soc 分支（S2 验收）
      IBr 单点 + 50 fs ISH
  4.2 QMMM 梯度合并（S1.5 验收）
      hydrated formaldehyde 10 fs
  4.3 ADAPTIVE 切换（S4 验收）
      NH3 切换 trajectory

Tier 5 — 性能/稳定性
  5.1 常驻性 — setup_interface 调用次数 = 1
  5.2 每步开销 benchmark
  5.3 MPI 多 rank 测试（若启用 MPI）
  5.4 长轨迹稳定性 — 1 ps 不泄漏 Python 对象
```

### 9.2 电子结构后端测试矩阵

| Backend | 物理体系 | 方法 | Tier 3.1 | Tier 3.2 | 特殊验证 |
|---|---|---|:-:|:-:|---|
| `ANALYTICAL` | 2-state 1D Tully | 解析 | ✓ | ✓ | 通路 smoke |
| `LVC` | pyrazine 4-mode | LVC | ✓ | ✓ | FAST 常驻 |
| `ORCA` | H2O | SA3-CASSCF(6,5) | ✓ | ✓ | NAC 符号 |
| `ORCA` | 乙烯 | TDDFT(PBE0) | ✓ | ✓ | 激发态梯度 |
| `ORCA` + SOC | IBr | SA-CASSCF + SOC | ✓ | ✓ | Vc/Ec/Tc 流 |
| `MOLCAS` | CH2O | SS-CASSCF(4,4) | ✓ | ✓ | 多重态排序 |
| `MOLCAS` + SOC | IBr | CASSCF+RASSI-SO | ✓ | ✓ | SOC matrix 对拍 |
| `TURBOMOLE` | 乙烯 | ADC(2) | ✓ | ✓ | 不同的约定 |
| `MNDO` | 2-state 小分子 | OM3 | ✓ | ✓ | 和 PSiNad 原生对拍 |
| `GAUSSIAN` | H2CO | TDDFT | ✓ | — | 单点即可 |
| `PYSCF` | H2O | CASSCF | ✓ | — | 纯 Python QC |
| `SPAINN` | CH2NH | schnetpack2 NN | ✓ | ✓ | 常驻验证 |
| `SCHNARC` | CH2NH | schnetpack1 NN | ✓ | — | 另一 NN 栈 |
| `QMMM(ORCA+OPENMM)` | 水合 CH2O | CASSCF/MM | ✓ | ✓ | MM 梯度合并 |
| `ADAPTIVE(ORCA+SPAINN)` | NH3 | QC↔NN | — | ✓ | 切换稳定性 |

"—" 表示该组合可选，不是阻断项。

### 9.3 Level A 单点对拍的构造方法

每个 backend 的单点测试在 `tests/example_parm/SHARC/<backend>/single_point/` 下：

```
single_point/
    input.toml                    # PSiNad 输入，运行 1 步即退出
    <BACKEND>.template            # SHARC 原生，和 sharc4 example 共享
    <BACKEND>.resources
    QM.in                         # 固定的几何
    reference.QMout               # sharc4/bin/driver.py 产生的参考
    compare.py                    # 比较 PSiNad 产生的 interface.ds vs reference
```

`compare.py` 逻辑：

```python
# 读 PSiNad DataSet dump
ds = read_ds("mdout/interface.ds")
ref = read_sharc_qmout("reference.QMout")
for key in ["eig", "grad", "nacdr", "dm"]:
    assert np.allclose(ds[key], ref[key], atol=1e-10, rtol=1e-10), key
```

参考数据一次生成后 checked in；backend 或 SHARC 版本变动需重生成。

### 9.4 Level B 短轨迹对拍的判据

```python
def compare_trajectory(psnd_out, sharc_out):
    # 1. 能量守恒
    assert abs(psnd_out.etot[-1] - psnd_out.etot[0]) < 1e-4  # ab-initio
    #  LVC / NN 更严: < 1e-6
    # 2. 末态占据
    assert np.allclose(psnd_out.occ[-1], sharc_out.occ[-1], atol=0.02)
    # 3. 总轨迹 RMSD
    rmsd = np.sqrt(((psnd_out.x - sharc_out.x) ** 2).mean())
    assert rmsd < 0.01  # Bohr
```

注意 surface hopping 的随机性：用**同一随机种子**（PSiNad 的 `solver.seed` 和 SHARC 的 random seed 对齐），否则概率性行为无法逐步对拍。如果对不上种子，降级到分布级别（100 条轨迹的态占据分布）。

### 9.5 性能回归（Tier 5）

建立 baseline 后写入 `tests/perf/SHARC_baseline.json`，CI 里每次跑回归：

```json
{
  "ANALYTICAL_1step_ms": 2.0,
  "LVC_1step_ms":        4.0,
  "SPAINN_1step_ms":     15.0,
  "ORCA_1step_s":        8.0,     // 不含 QC，仅协议开销 < 10 ms
  "QMMM_1step_s":        12.0
}
```

回归判据：性能退化超过 20% 告警。

---

## 10. 风险登记

| 风险 | 严重度 | 缓解 |
|---|---|---|
| pybind11 + PSiNad MPI 构建 GIL/interpreter 冲突 | 高 | S0 就加一个 MPI 2-rank smoke test；每 rank 独立 `scoped_interpreter` |
| SHARC 版本 drift（我们开发期间 SHARC 侧 API 变） | 中 | `tests/perf/sharc_version.txt` 记录 git sha；升级 SHARC 前先跑完整 Tier 3 |
| numpy 2.0 API breakage | 中 | CMake 里 pin `Python3::NumPy >= 1.22 AND < 2.0`，S0 验证后放宽 |
| 相位双翻（PSiNad `track_nac_sign` + SHARC `Adjust_phases`） | 中 | 在 `Model_SHARC_Interface` 分支禁用 `track_nac_sign`，让 SHARC 负责（feature 加 `phases`）；加 unit test 验证两帧 NAC 连续性 |
| MOLCAS / TURBOMOLE 多重态排序与 PSiNad `Dimension::F` 展平顺序不一致 | 中 | 在 `setup_mol` 阶段一次建立 `sharc_idx ↔ psnd_idx` 映射表，每次 getQMout 按表拷 |
| QMMM link atom / point charge 单位约定 | 中 | S1.5 单独写 `tests/SHARC/unit/test_qmmm_units.cpp`，对一个已知解析的 3 原子系统 check |
| Python 解释器在 `finalizeKernel_impl` 中 `Py_Finalize` 导致二次 Model 实例化 crash | 低 | 用 `Py_IsInitialized()` 守护，永不 Finalize；进程退出时由 OS 回收 |
| SHARC_ADAPTIVE 切换瞬间能量跳变 | 低 | template 层面设更大的不确定性阈值，或在切换点插入 1 步 micro-averaging（但这是 SHARC 的事，不是我们的） |

---

## 11. 进度跟踪

| 阶段 | 目标完成 | Commit prefix | Reference tests 通过 |
|---|---|---|---|
| S0 | Week 1 | `SHARC S0:` | Tier 1 全通 + Tier 2.1–2.4 |
| S1 | Week 2-3 | `SHARC S1:` | + Tier 3.1/3.2 对 ORCA/MOLCAS/TURBOMOLE/MNDO |
| S1.5 | Week 3 | `SHARC S1.5:` | + Tier 4.2 |
| S2 | Week 4 | `SHARC S2:` | + Tier 4.1 |
| S3 | Week 5 | `SHARC S3:` | + Tier 3 对 SPAINN/SCHNARC + Tier 5.1/5.2 |
| S4 | Week 6 | `SHARC S4:` | + Tier 4.3 |

每阶段结束写一个短 recap 到 `docs/dev/sharc_interface_progress.md`（即用即弃，不入主干文档）。

---

## 12. 明确不在当前范围的事（Non-goals）

以下是本次迁移刻意**延后**的事项。这里列出来是为了防止实现阶段被这些话题带偏。

| 延后项 | 延后原因 | 将来何时重开 |
|---|---|---|
| **单进程多轨迹 ensemble**（FAST 后端跨轨迹复用模型权重） | 当前阶段 M=1；先把 1 条轨迹跑对再谈跑多 | 如果 NN 后端实际工作流出现 M>1，再加进程级 `s_fast_instance_cache` + `std::mutex`。设计位点已在 § 1 标注 |
| **OpenMP / 线程级轨迹并行** | 先保证 pybind11 GIL 单线程语义正确 | 要加时跟多轨迹一起做 |
| **Python 解释器跨 Model 实例复用**（多个 SHARC Model 并存，如 ECI 嵌套） | 当前只有一个 Model_SHARC_Interface 实例 | S4 做 SHARC_ECI 时讨论 |
| **restart 文件交换格式** | 先走 SHARC 的 `create_restart_files` 原样落盘 | 后续如果想和 PSiNad 自己的 restart 打通再说 |

判断当前任务是否越界：如果某个 PR 开始往 `s_fast_instance_cache` 这类静态状态里加东西，应该立刻停下问一句"这是 S0–S1 范围的事吗"。
