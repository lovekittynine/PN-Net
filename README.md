# PN-Net
reproduce paper PN-Net:Conjoined Triplet Deep Netwprk for Learning Local Image Descriptor
# 有2点需要声明
# There are two important detail needed to be clarified
## 1 In author raw Lua code work ,I couldn't explictly find there is a batchnormlization.Maybe I dont konw how to use Lua and Torch
## 1 作者开源的Lua源码中没有显示的发现作者使用的BN,但是在复现中使用了BN
### The reason to use BN
Because in the procedure of optimization,there is always a gradient explode and thus cant continue to training,I dont kown why .I provide two solutions:First,use gradient clip in the train code which i comment it as i have used BN,once use BN,there is no optimization problem.Second,use BN

# I only do training on Brown Dataset Liberty 
# testing on the other two subsets Notredame and Yosemite
# results
