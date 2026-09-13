### 2026-09-13 20:23 Agent:Codex - 迁移 Eigen 实验并删除备份目录

# 需求
将 _bak/eigen.cpp 有用内容移入 main.cpp，然后删除 _bak。

# 结论
- 修改：main.cpp 用模板统一 float/double 的 32×32 乘法与 132×132 ABAt；重置相同随机种子。增加三对角稀疏矩阵乘法及校验值、非零元素数；输出 Eigen 版本、对齐值和编译时 SIMD 信息。
- 删除：_bak/eigen.cpp、_bak/rpiz-cli.c、_bak/rpiz-gtk.c 及空目录。未迁移无效标量循环、平台专属 CPUID 和调试输出。
- 验证：g++ -m64 与 aarch64-linux-gnu-g++ -march=armv8-a，均以 -std=c++11 -Wall -Wextra -Wformat=2 -fsyntax-only 检查 main.cpp，通过；存在 Eigen deprecated-copy 警告。未完整编译或运行，x86/ARMv7 未验证。
- 状态：迁移及删除完成，四架构 Makefile 无改动；ABAt/稀疏乘法为单次计时且包含结果分配开销。

### 2026-09-13 20:18 Agent:Codex - 整理入口与合并多架构 Makefile

# 需求
整理简易 CPU 测试程序，检查问题，合并 Makefile 并保留 x86/x64/armv7/arm64。

# 结论
- 修改：makefile 合并架构规则，删除两个 ARM 配置文件；输出按架构隔离，修正真实链接目标，增加头文件依赖及伪目标，允许工具链覆盖。main.cpp 清理无用代码，检查绑核/时钟错误，使用 PRId64，随机矩阵结果输出校验值。.gitignore 增加 /_temp/。
- 验证：五种 ARCH（含 native）make -n 通过，非法 ARCH 被拒绝；Linux x64、ARM64 入口语法检查通过，Eigen 存在既有 deprecated-copy 警告。x86 检查因缺少 bits/libc-header-start.h 失败；ARMv7 编译器缺失。未完整编译、链接或运行基准。
- 状态：代码与构建整理完成；rpiz 第三方内存管理问题只记录到 todo，未修改。详细构建方法见 docs/AGENTS.md。
