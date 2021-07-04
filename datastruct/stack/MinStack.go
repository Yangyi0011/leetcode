package main

import (
	"fmt"
)

/* 
========================== 1、最小栈==========================
	设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。
    push(x) —— 将元素 x 推入栈中。
    pop() —— 删除栈顶的元素。
    top() —— 获取栈顶元素。
	getMin() —— 检索栈中的最小元素。
	
	* Your MinStack object will be instantiated and called as such:
	* obj := Constructor();
	* obj.Push(val);
	* obj.Pop();
	* param_3 := obj.Top();
	* param_4 := obj.GetMin();

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/queue-stack/g5l7d/
*/
type MinStack struct {
	Data []int
	MinData []int
}

func Constructor() MinStack {
	return MinStack{Data : make([]int, 0), MinData : make([]int, 0)}
}

// 以空间换时间的方式，用另一个数组保存最小值数据，最小值总存在数组最后
func (this *MinStack) Push(x int)  {
	this.Data = append(this.Data, x)
	if len(this.MinData) == 0 {
		this.MinData = append(this.MinData, x)
	} else if x < this.MinData[len(this.MinData) - 1] {
		this.MinData = append(this.MinData, x)
	} else {
        // 仅仅只是把最小值放到最后而已，没有完整记录整个 stack 元素
		this.MinData = append(this.MinData, this.MinData[len(this.MinData) - 1])
	}
}

func (this *MinStack) Pop()  {
	length := len(this.Data)
	this.Data = this.Data[: length - 1]
	this.MinData = this.MinData[: length - 1]
}

func (this *MinStack) Top() int {
	return this.Data[len(this.Data) - 1]
}

func (this *MinStack) GetMin() int {
	return this.MinData[len(this.MinData) - 1]
}

/* 
========================== 2、单调栈（栈底到栈顶单调递减） =========================
	设计一个栈，保证栈元素按从底部到顶部是单调递减的。

	单调递减栈的应用：
		栈顶元素出栈时，一定是遇到了第一个比它大的元素，该元素即将入栈。
*/
type StackNode struct {
	Val int
	Prev *StackNode
	Next *StackNode
}

type Stack struct {
	// 栈容量
	Capacity int
	// 栈元素个数
	Count int
	// 链表头部、尾部
	Head, Tail *StackNode
}

/** initialize your data structure here. */
func Constructor() *Stack {
	return &Stack{
		// -1 表示无上限
		Capacity: -1,
		Count: 0,
		Head: nil,
		Tail: nil,
	}
}
// 栈是否为空
func (this *Stack) IsEmpty() bool {
	return this.Tail == nil
}
// 栈是否已满
func (this *Stack) IsFull() bool {
	if this.Capacity < 0 {
		return false
	}
	return this.Count == this.Capacity
}

// 压栈，压栈时确保最小元素在栈顶
func (this *Stack) Push(val int)  {
	if this.IsFull() {
		return
	}
	// 创建节点
	node := &StackNode{
		Val: val,
	}
	if this.IsEmpty() {
		this.Head = node
		this.Tail = node
		this.Count ++
		return
	}
	// val 比栈中的所有元素的值都大
	if val > this.Head.Val {
		node.Next = this.Head
		this.Head.Prev = node
		this.Head = node
		return
	}
	// Val 比栈中的所有元素的值都小
	if val < this.Tail.Val {
		node.Prev = this.Tail
		this.Tail.Next = node
		this.Tail = node
		return
	}
	// 把压栈元素移动到合适的位置
	p := this.Tail
	// 因为提前处理了 val 比栈中所有元素的值都大的情况
	// 所以这里不会存在 p == nil 的情况
	for val > p.Val {
		p = p.Prev
	}
	node.Next = p.Next
	if p.Next != nil {
		p.Next.Prev = node
	}
	p.Next = node
	node.Prev = p
}
// 弹出栈顶元素
func (this *Stack) Pop()  {
	if this.IsEmpty() {
		return
	}
	p := this.Tail.Prev
	this.Tail.Prev = nil
	p.Next = nil
	this.Tail = p
}
// 获取栈顶元素
func (this *Stack) Top() int {
	if this.IsEmpty() {
		return 0
	}
	return this.Tail.Val
}

func (stack Stack) String() string {
	res := ""
	p := stack.Head
	for p != nil {
		res += fmt.Sprintf("(%v)->", p.Val)
		p = p.Next
	}
	return res
}

func main() {
	stack := Constructor()
	fmt.Println(stack)
	stack.Push(-2)
	stack.Push(0)
	stack.Push(-3)
	fmt.Println(stack)
	stack.Pop()
	fmt.Println(stack)
	fmt.Println(stack.Top())
}