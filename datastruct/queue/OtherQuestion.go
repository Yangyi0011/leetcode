package queue


/* 
	关于队列的常见问题
*/

/* 
========================== 1、用队列实现栈 =========================
请你仅使用两个队列实现一个后入先出（LIFO）的栈，并支持普通队列的全部四种
操作（push、top、pop 和 empty）。
实现 MyStack 类：
    void push(int x) 将元素 x 压入栈顶。
    int pop() 移除并返回栈顶元素。
    int top() 返回栈顶元素。
    boolean empty() 如果栈是空的，返回 true ；否则，返回 false 。

注意：
	你只能使用队列的基本操作 —— 也就是 push to back、peek/pop from 
	front、size 和 is empty 这些操作。
	你所使用的语言也许不支持队列。 你可以使用 list （列表）或者 deque
	（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。

示例：
输入：
["MyStack", "push", "push", "top", "pop", "empty"]
[[], [1], [2], [], [], []]
输出：
[null, null, null, 2, 2, false]

解释：
MyStack myStack = new MyStack();
myStack.push(1);
myStack.push(2);
myStack.top(); // 返回 2
myStack.pop(); // 返回 2
myStack.empty(); // 返回 False

提示：
    1 <= x <= 9
    最多调用100 次 push、pop、top 和 empty
    每次调用 pop 和 top 都保证栈不为空

进阶：你能否实现每种操作的均摊时间复杂度为 O(1) 的栈？换句话说，执行 n 个操作的总时间复杂度 O(n) ，尽管其中某个操作可能需要比其他操作更长的时间。你可以使用两个以上的队列。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/queue-stack/gw7fg/
*/
/* 
	方法一：双队列
	思路：
		我们定义 q1、q2 两个队列。
		1、遇到 push 操作时，我们先把 x 入队到 q2，再把 q1 的所有元素
			出队到 q2，最后交换 q1、q2，此时 q1 的元素即为栈元素，
			q1 的头部即为栈顶，尾部即为栈底。
		2、遇到 top、pop操作时，直接从 q1 头部取元素进行 top 或 pop
		3、因为所有元素都在 q1 中，我们判空时只需要判断 q1 即可。
	时间复杂度：O(n)
		n 是操作次数，empty、pop、peek 耗时 o(1)，而每一次 push 都耗
		时 o(n)，故总的均摊时间复杂度为 o(n)
	空间复杂度：O(n)
		n 是操作次数，n 次 push 会让栈中存在 n 个元素。
*/
type MyStack struct {
	q1 []int
	q2 []int
}

/** Initialize your data structure here. */
func Constructor() MyStack {
	return MyStack{
		q1: make([]int, 0),
		q2: make([]int, 0),
	}
}

// 把 q1 的元素倒入 q2
func (this *MyStack) Q1ToQ2() {
	for len(this.q1) > 0 {
		this.q2 = append(this.q2, this.q1[0])
		this.q1 = this.q1[1:]
	}
}

/** Push element x onto stack. */
func (this *MyStack) Push(x int)  {
	this.q2 = append(this.q2, x)
	this.Q1ToQ2()
	// 交换 q1、q2
	this.q1, this.q2 = this.q2, this.q1
}

/** Removes the element on top of the stack and returns that element. */
func (this *MyStack) Pop() int {
	x := this.q1[0]
	this.q1 = this.q1[1:]
	return x
}

/** Get the top element. */
func (this *MyStack) Top() int {
	return this.q1[0]
}

/** Returns whether the stack is empty. */
func (this *MyStack) Empty() bool {
	return len(this.q1) == 0
}

/**
 * Your MyStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(x);
 * param_2 := obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.Empty();
 */

 /* 
	方法二：单队列
	思路：
		我们定义一个队列 queue。
		遇到 push 操作时：
			我们先记录当前队列元素个数 n，然后把 x 入队，最后再把前 n 个
			元素依次出队、入队（不包括新入队的 x），此时 x 处在的位置即为
			队列的头部，而此时队列的头部即为栈顶，队列的尾部即为栈底。
		遇到 pop、top 操作时：
			我们直接从 queue 的头部取元素进行 top、pop 即可。
	时间复杂度：O(n)
		n 是操作次数，除了 push 操作耗时为 O(n)，其余操作都是 O(1)
	空间复杂度：O(n)
		n 是操作次数，n 次 push 会让栈中存在 n 个元素。
 */
type MyStack struct {
	queue []int
}

/** Initialize your data structure here. */
func Constructor() MyStack {
	return MyStack{
		queue: make([]int, 0),
	}
}

/** Push element x onto stack. */
func (this *MyStack) Push(x int)  {
	n := len(this.queue)
	this.queue = append(this.queue, x)
	// 前 n 个元素出队、入队
	for i := 0; i < n; i ++ {
		this.queue = append(this.queue, this.queue[0])
		this.queue = this.queue[1:]
	}
}

/** Removes the element on top of the stack and returns that element. */
func (this *MyStack) Pop() int {
	x := this.queue[0]
	this.queue = this.queue[1:]
	return x
}

/** Get the top element. */
func (this *MyStack) Top() int {
	return this.queue[0]
}

/** Returns whether the stack is empty. */
func (this *MyStack) Empty() bool {
	return len(this.queue) == 0
}

/**
 * Your MyStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(x);
 * param_2 := obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.Empty();
 */