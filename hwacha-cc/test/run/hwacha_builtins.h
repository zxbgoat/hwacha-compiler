// Declarations hwacha-cc understands beyond OpenCL C 1.2: work-group reductions (OpenCL 2.0 names).
// Scope: one stripmine group = the work-group (size via reqd_work_group_size or hwacha_group_size).
#ifndef HWACHA_BUILTINS_H
#define HWACHA_BUILTINS_H
float __attribute__((overloadable)) work_group_reduce_add(float);
float __attribute__((overloadable)) work_group_reduce_min(float);
float __attribute__((overloadable)) work_group_reduce_max(float);
int   __attribute__((overloadable)) work_group_reduce_add(int);
int   __attribute__((overloadable)) work_group_reduce_min(int);
int   __attribute__((overloadable)) work_group_reduce_max(int);
#endif
