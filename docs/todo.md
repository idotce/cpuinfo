# 待办

- 用户授权完整编译后，在 x86、x64、ARMv7、ARM64 实际构建并运行验证；当前环境缺少 x86 32 位开发头文件及 ARMv7 交叉编译器。
- 经用户授权修改第三方源码后，处理 `src/util.c`：`strlist_free` 未释放 `list->strs`；`strlist_new` 未检查 malloc 失败；`strlist_add_w` 在分配成功前增加 count，失败后状态无效；`get_file_contents` 分配回退分支写 `buff[fs+1]` 而非 `buff[fs]`，且会在整页边界越界。以上为代码检查发现，未进行运行时复现。
- 进一步审核 rpiz 固定缓冲区和 CPU 数量边界、字段值所有权；当前不视为已经完成全面安全审计。
- 内置 Eigen 在现有 GCC 下出现 deprecated-copy 警告，本次未升级或修改依赖。
