package main

/*
============== 剑指 Offer 09. 用两个栈实现队列 ==============
用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和
deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。
(若队列中没有元素，deleteHead 操作返回 -1 )

示例 1：
输入：
["CQueue","appendTail","deleteHead","deleteHead"]
[[],[3],[],[]]
输出：[null,null,3,-1]

示例 2：
输入：
["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
[[],[],[5],[2],[],[]]
输出：[null,-1,null,null,5,2]

提示：
    1 <= values <= 10000
    最多会对 appendTail、deleteHead 进行 10000 次调用

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof
*/
/**
 * Your CQueue object will be instantiated and called as such:
 * obj := Constructor();
 * obj.AppendTail(value);
 * param_2 := obj.DeleteHead();
 */

/*
	方法一：inStack、outStack
	思路：
		使用两个栈 inStack、outStack，inStack 负责队列元素的入队，outStcak
		负责队列元素的出队。
		当执行入队操作时，直接把入队元素 push 进 inStack，执行出队操作时，
		如果 outStack 为空，需要先把 inStack 的元素全部倒进 outStack，
		再把 outStack 的栈顶元素 pop 出队，否则直接把 outStack 的栈顶元素
		pop 出队。
	时间复杂度：O(1)
		对于插入和删除操作，时间复杂度均为 O(1)。插入不多说，对于删除操作
		，虽然看起来是 O(n)的时间复杂度，但是仔细考虑下每个元素只会
		「至多被插入和弹出 outStack 一次」，因此均摊下来每个元素被删除的时
		间复杂度仍为 O(1)。
	空间复杂度：O(n)
		需要使用两个栈存储已有的元素。
*/
type CQueue struct {
	inStack, outStack []int
}

func Constructor() CQueue {
	return CQueue{inStack: []int{}, outStack: []int{}}
}

func (this *CQueue) AppendTail(value int) {
	this.inStack = append(this.inStack, value)
}

func (this *CQueue) IsEmpty() bool {
	return len(this.inStack) == 0 && len(this.outStack) == 0
}

func (this *CQueue) DeleteHead() int {
	if this.IsEmpty() {
		return -1
	}
	if len(this.outStack) <= 0 {
		for len(this.inStack) > 0 {
			this.outStack = append(this.outStack, this.inStack[len(this.inStack)-1])
			this.inStack = this.inStack[:len(this.inStack)-1]
		}
	}
	val := this.outStack[len(this.outStack)-1]
	this.outStack = this.outStack[:len(this.outStack)-1]
	return val
}
