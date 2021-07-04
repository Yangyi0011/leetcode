package stack

/* 
========================== 1、设计栈=========================
*/
type Stack struct {
	// 栈元素个数
	count int
	// 栈容量
	capacity int
	// 栈顶
	top int
	// 栈元素列表
	values []int
}

// 构造函数
func Constructor(k int) *Stack {
	return &Stack {
		count: 0,
		capacity: k,
		top: 0,
		values: make([]int, k),
	}
}

// 栈是否已满
func (this *Stack) IsFull() bool {
	return this.count == this.capacity
}

// 栈是否为空
func (this *Stack) IsEmpty() bool {
	return this.count == 0
}

// 压栈，成功则返回刚刚压栈的元素，失败返回 0
func (this *Stack) Push(val int) (int, bool) {
	if this.IsFull() {
		reutrn 0, false
	}
	this.values[this.top] = val
	this.count ++
	this.top ++
	return val, true
}

// 出栈，成功则返回栈顶元素，失败返回 0
func (this *Stack) Pop() (int, bool) {
	if this.IsEmpty() {
		return 0, false
	}
	val := this.values[this.top-1]
	this.count --
	this.top --
	return val, true
}

// 查看栈顶元素，成功返回栈顶元素，失败返回 -1
func (this *Stack) Top() (int, bool) {
	if this.IsEmpty() {
		return 0, false
	}
	return this.values[this.top-1]
}

/* 
========================== 2、有效的括号=========================
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字
符串是否有效。

有效字符串需满足：
    左括号必须用相同类型的右括号闭合。
    左括号必须以正确的顺序闭合。

示例 1：
输入：s = "()"
输出：true

示例 2：
输入：s = "()[]{}"
输出：true

示例 3：
输入：s = "(]"
输出：false

示例 4：
输入：s = "([)]"
输出：false

示例 5：
输入：s = "{[]}"
输出：true

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/queue-stack/g9d0h/
*/
/* 
	方法一：栈
	思路：
		把字符串拆分成字符数组，将字符数组的字符按顺序进行入栈和出栈：
			1、遇到左括号直接入栈
			2、遇到右括号：
				（1）栈为空，直接入栈。
				（2）栈不为空，如果栈顶元素是同类型的左括号，则栈顶元素出栈，
					之后处理下一个字符。否则直接入栈
			3、最后判断栈是否为空来确认字符串是否合法
	时间复杂度：O(n)
		n 是字符串的长度，我们需要对每一个字符都进行入栈/出栈操作。
	空间复杂度：O(n)
		我们需要一个字符数组来存储栈元素，最坏情况下每一个字符都需要入栈。
*/
func isValid(s string) bool {
	n := len(s)
	if n == 0 {
		return true
	}
	stack := make([]byte, 0)
	for i := 0; i < n; i ++ {
		// 栈为空，直接入栈
		if len(stack) == 0 {
			stack = append(stack, s[i])
			continue
		}
		switch s[i] {
			// 左括号直接入栈
			case '(', '[', '{':
				stack = append(stack, s[i])
			case ')':
				if stack[len(stack) - 1] == '(' {
					stack = stack[:len(stack) - 1]
				} else {
					stack = append(stack, s[i])
				}
			case ']':
				if stack[len(stack) - 1] == '[' {
					stack = stack[:len(stack) - 1]
				} else {
					stack = append(stack, s[i])
				}
			case '}':
				if stack[len(stack) - 1] == '{' {
					stack = stack[:len(stack) - 1]
				} else {
					stack = append(stack, s[i])
				}
			default:
		}
	}
	return len(stack) == 0
}

/*
========================== 3、每日温度 =========================
请根据每日 气温 列表，重新生成一个列表。对应位置的输出为：要想观测到更
高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 
0 来代替。
例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，
你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。
提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的均为华氏度，都是
在 [30, 100] 范围内的整数。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/queue-stack/genw3/
*/
/* 
	方法一：单调栈（栈低到栈顶单调递减）
	思路：
		可以维护一个存储下标的单调栈，从栈底到栈顶的下标对应的温度列表中
		的温度依次递减。如果一个下标在单调栈里，则表示尚未找到下一次温度
		更高的下标。
		正向遍历温度列表。对于温度列表中的每个元素 T[i]，如果栈为空，则
		直接将 i 进栈，如果栈不为空，则比较栈顶元素 prevIndex 对应的温
		度 T[prevIndex] 和当前温度 T[i]，如果 T[i] > T[prevIndex]，则
		将 prevIndex 移除，并将 prevIndex 对应的等待天数赋为 
		i - prevIndex，重复上述操作直到栈为空或者栈顶元素对应的温度小于
		等于当前温度，然后将 i 进栈。

		为什么可以在弹栈的时候更新 ans[prevIndex] 呢？因为在这种情况下，
		即将进栈的 i 对应的 T[i] 一定是 T[prevIndex] 右边第一个比它大的
		元素，试想如果 prevIndex 和 i 有比它大的元素，假设下标为 j，那
		么 prevIndex 一定会在下标 j 的那一轮被弹掉。

		由于单调栈满足从栈底到栈顶元素对应的温度递减，因此每次有元素进栈
		时，会将温度更低的元素全部移除，并更新出栈元素对应的等待天数，这
		样可以确保等待天数一定是最小的。
	时间复杂度：O(n)
		其中 n 是温度列表的长度。正向遍历温度列表一遍，对于温度列表中的
		每个下标，最多有一次进栈和出栈的操作。
	空间复杂度：O(n)
		其中 n 是温度列表的长度。需要维护一个单调栈存储温度列表中的下标。
*/
func dailyTemperatures(T []int) []int {
	n := len(T)
	// 结果集
	ans := make([]int, n)
	stack := make([]int, 0)
	for i := 0; i < n; i ++ {
		// 栈不为空，对比栈顶下标对应的温度和当前温度
		for len(stack) > 0 && T[i] > T[stack[len(stack) - 1]] {
			prevIndex := stack[len(stack) - 1]
			// 计算栈顶下标对应的温度需要等待的天数
			ans[prevIndex] = i - prevIndex
			// 弹出栈顶
			stack = stack[:len(stack) - 1]
		}
		// 栈为空时会跳过循环，直接把当前温度的下标入栈
		// 当前温度的下标进栈
		stack = append(stack, i)
	}
	return ans
}

/* 
========================== 4、逆波兰表达式求值 =========================
根据 逆波兰表示法，求表达式的值。
有效的算符包括 +、-、*、/ 。每个运算对象可以是整数，也可以是另一个逆波
兰表达式。

说明：
    整数除法只保留整数部分。
	给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存
	在除数为 0 的情况。

示例 1：
输入：tokens = ["2","1","+","3","*"]
输出：9
解释：该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9

示例 2：
输入：tokens = ["4","13","5","/","+"]
输出：6
解释：该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6

示例 3：
输入：tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
输出：22
解释：
该算式转化为常见的中缀算术表达式为：
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/evaluate-reverse-polish-notation
*/
/* 
	方法一：栈
	思路：
		把输入字符进行入栈和出栈操作，遇到数字直接入栈，遇到运算符则
		从栈中弹出两个元素进行计算，其计算结果再入栈。
	时间复杂度：O(n)
		n 是字符数组的元素个数，我们需要遍历整个字符数组，每个字符最多
		只有一次入栈和出栈操作。
	空间复杂度：O(n)
		n 是字符数组的元素个数，我们需要用一个栈来存储数字字符和
		计算的中间结果。
*/
func evalRPN(tokens []string) int {
	n := len(tokens)
	if n == 0 {
		return 0
	}
	stack := make([]int, 0)
	for i := 0; i < n; i ++ {
		c := tokens[i]
		// 是字符则需要计算
		if c == "+" || c == "-" || c == "*" || c == "/" {
			// 弹出两个数字进行计算
			a := stack[len(stack) -1]
			b := stack[len(stack) -2]
			stack = stack[:len(stack) -2]
			// 计算结果
			var res int
			switch c {
				case "+":
					res = b + a
				case "-":
					res = b - a
				case "*":
					res = b * a
				case "/":
					res = b / a
			}
			// 计算结果入栈
			stack = append(stack, res)
			continue
		}
		// 是数字字符，则转为数字直接入栈
		num, _ := strconv.Atoi(c)
		stack = append(stack, num)
	}
	// 最终栈中只有一个元素，即为最终结果
	return stack[0]
}