# 项目说明

- Linux CPU 信息展示与 Eigen 矩阵运算简易测试；依赖 `/proc`、`/sys` 和 Linux CPU affinity API。当前不支持 Windows/macOS 原生运行。
- `main.cpp` 是程序入口；`src/` 带 rpiz 第三方版权声明，`Eigen/` 是内置第三方依赖。未经明确同意不修改第三方源码。
- 唯一构建入口为根目录 `makefile`，使用 GNU Make 与 GCC/G++，无需 CMake。
- `make ARCH=x86`、`make ARCH=x64`、`make ARCH=armv7`、`make ARCH=arm64`；默认 `make` 使用本机编译器，输出 `out/native/cpuinfo`。显式架构输出 `out/<ARCH>/cpuinfo`。
- x86 需要 32 位开发库；ARMv7 默认 `arm-linux-gnueabihf-` 工具链，要求 ARMv7-A、NEON、hard-float；ARM64 默认 `aarch64-linux-gnu-` 工具链，使用 ARMv8-A。
- 可用 `CROSS_COMPILE=前缀` 或 `CC=... CXX=...` 指定工具链；ARM 本机编译可显式传 `CROSS_COMPILE=`。架构应与工具链一致。
- 默认静态链接，可传 `LDFLAGS=` 使用动态链接。`make clean ARCH=...` 仅清理对应架构；同一架构更换工具链或命令行编译参数前先清理该架构输出。
- `make -n ARCH=...` 只检查命令展开；完整编译需用户明确要求。
- 基准已改用随机矩阵并输出校验值，32×32 循环含结果消费开销，132×132 ABAt 仍为单次测量、包含分配开销；不能将新旧耗时直接比较，也不是完整 CPU 性能评分。
- `main.cpp` 统一执行 float/double 稠密矩阵乘法、ABAt 和三对角稀疏矩阵乘法，输出校验值。稀疏乘法为单次计时、含结果分配；SIMD 输出表示编译时启用功能。`_bak/` 已按用户要求删除。
