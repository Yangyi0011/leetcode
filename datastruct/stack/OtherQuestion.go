package stack

/* 
	关于队列和栈的常见问题
*/

/* 
========================== 1、用栈实现队列 =========================
请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作
（push、pop、peek、empty）：
实现 MyQueue 类：
    void push(int x) 将元素 x 推到队列的末尾
    int pop() 从队列的开头移除并返回元素
    int peek() 返回队列开头的元素
    boolean empty() 如果队列为空，返回 true ；否则，返回 false

说明：
	你只能使用标准的栈操作 —— 也就是只有 push to top, peek/pop from 
	top, size, 和 is empty 操作是合法的。
	你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）
	来模拟一个栈，只要是标准的栈操作即可。
进阶：
	你能否实现每个操作均摊时间复杂度为 O(1) 的队列？换句话说，执行 n 
	个操作的总时间复杂度为 O(n) ，即使其中一个操作可能花费较长时间。

示例：
输入：
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]
输出：
[null, null, null, 1, 1, false]

解释：
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false

提示：
    1 <= x <= 9
    最多调用 100 次 push、pop、peek 和 empty
    假设所有操作都是有效的 （例如，一个空的队列不会调用 pop 或者 peek 操作）

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/queue-stack/gvtxe/
*/
/* 
	方法一：InStack、OutStack
	思路：
		定义 InStack 和 OutStack 两个栈。
		1、InStack 负责队列元素的 Push 操作，当有元素 x 需要 push 时，
			如果队列为空，直接把 x 压入 InStack 中，否则需要判断 
			OutStack 是否为空：
			（1）OutStack 不为空，需要把 OutStack 的元素一一弹出并压入
				InStack 中，最后再压入 x 元素
			（2）OutStack 为空，直接把 x 压入 InStack
		2、OutStack 负责队列元素 Pop 和 Peek 操作，当有元素要 Pop 或 Peek
			时，需要先判断 InStack 是否为空：
			（1）InStack 不为空，需要把 InStack 的元素一一弹出并压入
				OutStack 中，最后从 OutStack 中获取栈顶元素 Pop 或 Peek。
			（2）InStack 为空，直接从 OutStack 中获取栈顶元素 Pop 或 Peek。
	时间复杂度：O(n)
		n 是操作次数，其中 empty 的时间复杂度为 O(1)，push 为 O(n)，而
		pop、peek 的均摊时间复杂度为 O(1)，所以总的均摊时间复杂度为 O(n)。
	空间复杂度：O(n)
		n 是操作次数，当存在 n 次 push 时，队列中的元素个数为 n 个。
*/
type MyQueue struct {
	InStack []int
	OutStack []int
}
/** Initialize your data structure here. */
func Constructor() MyQueue {
	return MyQueue{
		InStack: make([]int, 0),
		OutStack: make([]int, 0),
	}
}
/** Push element x to the back of queue. */
func (this *MyQueue) Push(x int)  {
	if this.Empty() {
		this.InStack = append(this.InStack, x)
		return
	}
	for len(this.OutStack) > 0 {
		val := this.OutStack[len(this.OutStack) - 1]
		this.OutStack = this.OutStack[: len(this.OutStack) - 1]
		this.InStack = append(this.InStack, val)
	}
	this.InStack = append(this.InStack, x)
}
/** Removes the element from in front of queue and returns that element. */
func (this *MyQueue) Pop() int {
	if this.Empty() {
		return 0
	}
	for len(this.InStack) > 0 {
		val := this.InStack[len(this.InStack) - 1]
		this.InStack = this.InStack[: len(this.InStack) - 1]
		this.OutStack = append(this.OutStack, val)
	}
	outVal := this.OutStack[len(this.OutStack) - 1]
	this.OutStack = this.OutStack[: len(this.OutStack) - 1]
	return outVal
}
/** Get the front element. */
func (this *MyQueue) Peek() int {
	if this.Empty() {
		return 0
	}
	for len(this.InStack) > 0 {
		val := this.InStack[len(this.InStack) - 1]
		this.InStack = this.InStack[: len(this.InStack) - 1]
		this.OutStack = append(this.OutStack, val)
	}
	return this.OutStack[len(this.OutStack) - 1]
}
/** Returns whether the queue is empty. */
func (this *MyQueue) Empty() bool {
	return len(this.InStack) == 0 && len(this.OutStack) == 0
}

/**
 * Your MyQueue object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(x);
 * param_2 := obj.Pop();
 * param_3 := obj.Peek();
 * param_4 := obj.Empty();
 */

 /* 
	方法二：InStack、OutStack 优化
	思路：
		方法一中的 O(n) 时间复杂度明显不符合要求，那么我们该怎么优化呢？
		方法一的时间复杂度过高的主要问题在于 Push 时需要先从 OutStack 把
		元素倒入 InStack，而 Pop 和 Peek 时，需要先从 InStack 把元素倒入
		OutStack，由此造成了 O(n) 的均摊时间复杂度。我们能不能在 Push 与
		Pop、Peek 时，只倒一次呢？
		答案是可以的，我们可以这样做：
			1、Push 时，直接把 x 压入 InStack 中。
			2、Pop 和 Peek 时，先判断 OutStack 是否为空：
				（1）OutStack 不为空，直接从 OutStack 取栈顶元素进行 Pop
					或 Peek。
				（2）OutStack 为空，需要先把 InStack 的元素倒入 OutStack
					中，再从 OutStack 取栈顶元素进行 Pop 或 Peek。
		这样如何能保证操作是正确的呢？
		Push 时元素被压入 InStack 中，Pop 或 Peek 时，如果 OutStack 为空
		则先把 InStack 的元素倒入 OutStack 中，再从 OutStack 中取栈顶元素
		进行 Pop 或 Peek，此时 Pop 或 Peek 出去的元素必然是最先 Push 进
		InStack 的那个元素。而如果 OutStack 不为空，我们直接取 OutStack 的
		栈顶元素进行 Pop 和 Peek，此时 Pop 或 Peek 出去的元素也必然是最先
		Push 进 InStack 的那个元素，后面 Push 的元素还保留在 InStack 中，
		所以操作是正确的。
	时间复杂度：O(n)
		n 是操作次数，其中 empty、Push 的时间复杂度为 O(1)，
		pop、peek 的均摊时间复杂度为 O(1)，所以总的均摊时间复杂度为 O(1)。
	空间复杂度：O(n)
		n 是操作次数，当存在 n 次 push 时，队列中的元素个数为 n 个。
*/
type MyQueue struct {
	InStack []int
	OutStack []int
}
/** Initialize your data structure here. */
func Constructor() MyQueue {
	return MyQueue{
		InStack: make([]int, 0),
		OutStack: make([]int, 0),
	}
}
/** Push element x to the back of queue. */
func (this *MyQueue) Push(x int)  {
	// 直接入栈，不需要管 OutStack
	this.InStack = append(this.InStack, x)
}
// 把 InStack 的元素倒入 OutStack 中
func (this *MyQueue) InToOut() {
	for len(this.InStack) > 0 {
		val := this.InStack[len(this.InStack) - 1]
		this.InStack = this.InStack[: len(this.InStack) - 1]
		this.OutStack = append(this.OutStack, val)
	}
}
/** Removes the element from in front of queue and returns that element. */
func (this *MyQueue) Pop() int {
	if len(this.OutStack) == 0 {
		this.InToOut()
	}
	outVal := this.OutStack[len(this.OutStack) - 1]
	this.OutStack = this.OutStack[: len(this.OutStack) - 1]
	return outVal
}
/** Get the front element. */
func (this *MyQueue) Peek() int {
	if len(this.OutStack) == 0 {
		this.InToOut()
	}
	return this.OutStack[len(this.OutStack) - 1]
}
/** Returns whether the queue is empty. */
func (this *MyQueue) Empty() bool {
	return len(this.InStack) == 0 && len(this.OutStack) == 0
}

/* 
========================== 2、字符串解码 =========================
给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 
正好重复 k 次。注意 k 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括
号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如
不会出现像 3a 或 2[4] 的输入。

示例 1：
输入：s = "3[a]2[bc]"
输出："aaabcbc"

示例 2：
输入：s = "3[a2[c]]"
输出："accaccacc"

示例 3：
输入：s = "2[abc]3[cd]ef"
输出："abcabccdcdcdef"

示例 4：
输入：s = "abc3[cd]xyz"
输出："abccdcdcdxyz"

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/decode-string
*/
/* 
	方法一：栈
	思路：
		看到中括号匹配，我们很容易就能想到要用栈来处理。具体如下：
			我们把整个字符串拆分成字符数组，把数组元素一一入栈，
			入栈时如果遇到 ']' 就弹出栈顶元素，直到遇到 '[' 为止，
			把弹出的字符按弹出顺序倒序拼接（'['、']'除外）为 charStr，
			然后还要检查栈顶元素是否为数字，如果是数字也要跟着弹出，
			注意：数字可能会有多位，如 13 这种，所以需要多次检查，
			把弹出的数字也按弹出顺序倒序拼接为 numStr，之后把 numStr
			转为对应的数字 num，再按 num 次数重复拼接 charStr 得到 str，
			最后把 str 压入栈中。
			再重复上述过程，直到字符数组的元素完全入栈，此时栈中的元素即为
			解码结果，我们只需要把栈元素全部弹出按逆序拼接，就能得到答案。
	时间复杂度：O(n)
		n 为解码字符串的长度，解码字符串的字符至多需要入栈/出栈两次。
	空间复杂度：O(n)
		n 为解码字符串的长度，解码后的字符串整体字符都需要入栈一次。
*/
func decodeString(s string) string {
	n := len(s)
	if n == 0 {
		return s
	}
	stack := make([]string, 0)
	for i := 0; i < n; i ++ {
		if s[i] == ']' {
			// 弹出字符
			charStr := ""
			// 弹出来的是 string 类型
			tmp := stack[len(stack) - 1]
			for tmp != "[" {
				// 逆序拼接
				charStr = tmp + charStr
				stack = stack[: len(stack) - 1]
				tmp = stack[len(stack) - 1]
			}
			// 扔掉 "[""
			stack = stack[: len(stack) - 1]
			
			// 弹出数字
			numStr := ""
			tmp = stack[len(stack) - 1]
			for tmp[0] >= '0' && tmp[0] <= '9' {
				// 逆序拼接
				numStr = tmp + numStr
				stack = stack[: len(stack) - 1]
				// 这里可能会越界，需要注意
				if len(stack) == 0 {
					break
				}
				tmp = stack[len(stack) - 1]
			}
			// 把 numStr 转为数字
			num, _ := strconv.Atoi(numStr)
			// 按 num 次数拼接 charStr
			str := ""
			for i := 0; i < num; i ++ {
				str += charStr
			}

			// str 入栈
			stack = append(stack, str)
			continue
		}
		// 其他字符直接入栈
		stack = append(stack, string(s[i]))
	}
	// 拼接结果
	result := ""
	// 这里就不用弹出栈的方式了，直接当成切片遍历
	for _, v := range stack {
		result += v
	}
	return result
}

/* 
========================== 3、图像渲染 =========================
有一幅以二维整数数组表示的图画，每一个整数表示该图画的像素值大小，数值
在 0 到 65535 之间。
给你一个坐标 (sr, sc) 表示图像渲染开始的像素值（行 ，列）和一个新的颜色
值 newColor，让你重新上色这幅图像。
为了完成上色工作，从初始坐标开始，记录初始坐标的上下左右四个方向上像素值
与初始坐标相同的相连像素点，接着再记录这四个方向上符合条件的像素点与他们
对应四个方向上像素值与初始坐标相同的相连像素点，……，重复该过程。将所有有
记录的像素点的颜色值改为新的颜色值。
最后返回经过上色渲染后的图像。

示例 1:
输入: 
image = [[1,1,1],[1,1,0],[1,0,1]]
sr = 1, sc = 1, newColor = 2
输出: [[2,2,2],[2,2,0],[2,0,1]]
解析: 
1 1 1
1 1 0
1 0 1
在图像的正中间，(坐标(sr,sc)=(1,1)),
在路径上所有符合条件的像素点的颜色都被更改成2。
注意，右下角的像素没有更改为2，
因为它不是在上下左右四个方向上与初始点相连的像素点。

注意:
    image 和 image[0] 的长度在范围 [1, 50] 内。
    给出的初始点将满足 0 <= sr < image.length 和 0 <= sc < image[0].length。
    image[i][j] 和 newColor 表示的颜色值在范围 [0, 65535]内。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/queue-stack/g02cj/
*/
/* 
	方法一：递归DFS
	思路：
		我们把图像的所有像素点看作是一个以 image[sr][sc] 值连通的无向图，
		我们记录原始颜色的值 image[sr][sc]，然后从 image[sr][sc] 出发
		向 上下左右 四个方向按颜色值与 原始颜色相同的 路径发散，发散的同
		时把走过的路径的颜色改为新的颜色的值。
		注意：如果原颜色与新颜色相同，则不需要进行修改直接返回，否则会
			死循环。
	时间复杂度：O(mn)
		m、n 分别为 image 的行和列，最坏情况下我们需要把整幅图像的所有
		像素点都重新染色。
	空间复杂度：O(mn)
		m、n 分别为 image 的行和列，最坏情况下我们要递归处理所有的像素点，
		此时耗费的递归栈空间为 O(mn)。 
*/
func floodFill(image [][]int, sr int, sc int, newColor int) [][]int {
	// 记录原颜色的值
	oldColor := image[sr][sc]
	// 原颜色与新颜色相同，不用染色
	if newColor == oldColor {
		return image
	}
	m, n := len(image), len(image[0])
	var DFS func(i, j int)
	DFS = func(i, j int) {
		if i < 0 || i >= m || j < 0 || j >= n || image[i][j] != oldColor {
			return
		}
		// 染色
		image[i][j] = newColor
		// 向上下左右四个方向发散
		DFS(i - 1, j)
		DFS(i + 1, j)
		DFS(i, j - 1)
		DFS(i, j + 1)
	}
	DFS(sr, sc)
	return image
}

/* 
	方法二：BFS
	思路：
		能用 DFS 处理的问题都能用 BFS 处理，只需要借助于一个额外队列。
		我们记录原颜色的值，然后从 image[i][j] 开始处理，并把 
		image[i][j] 的邻居节点放入队列中，一一处理队列节点，直到队列为空。
	时间复杂度：O(mn)
		m、n 分别为 image 的行和列，最坏情况下我们需要把整幅图像的所有
		像素点都重新染色。
	空间复杂度：O(mn)
		其中 n 和 m 分别是二维数组的行数和列数。主要为队列的开销。
*/
func floodFill(image [][]int, sr int, sc int, newColor int) [][]int {
	oldColor := image[sr][sc]
	if oldColor == newColor {
		return image
	}
	m, n := len(image), len(image[0])
	queue := make([][]int, 0)
	// 处理起始点
	queue = append(queue, []int{sr, sc})
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		i, j := cur[0], cur[1]
		// 处理当前节点
		image[i][j] = newColor
		// 处理邻居节点
		if i - 1 >= 0 && image[i-1][j] == oldColor {
			queue = append(queue, []int{i - 1, j})
		}
		if i + 1 < m && image[i+1][j] == oldColor {
			queue = append(queue, []int{i + 1, j})
		}
		if j - 1 >= 0 && image[i][j - 1] == oldColor {
			queue = append(queue, []int{i, j - 1})
		}
		if j + 1 < n && image[i][j + 1] == oldColor {
			queue = append(queue, []int{i, j + 1})
		}
	}
	return image
}

/* 
========================== 4、01 矩阵 =========================
给定一个由 0 和 1 组成的矩阵，找出每个元素到最近的 0 的距离。
两个相邻元素间的距离为 1 。

示例 1：
输入：
[[0,0,0],
 [0,1,0],
 [0,0,0]]
输出：
[[0,0,0],
 [0,1,0],
 [0,0,0]]

示例 2：
输入：
[[0,0,0],
 [0,1,0],
 [1,1,1]]
输出：
[[0,0,0],
 [0,1,0],
 [1,2,1]]

提示：
    给定矩阵的元素个数不超过 10000。
    给定矩阵中至少有一个元素是 0。
    矩阵中的元素只在四个方向上相邻: 上、下、左、右。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/queue-stack/g7pyt/
*/
/* 
	方法一：BFS
	思路：
		对于矩阵中的每一个元素，如果它的值为 000，那么离它最近的 0 就是
		它自己。如果它的值为 1，那么我们就需要找出离它最近的 0，并且返回
		这个距离值。那么我们如何对于矩阵中的每一个 1，都快速地找到离它最
		近的 0 呢？
		我们不妨从一个简化版本的问题开始考虑起。假设这个矩阵中恰好只有一
		个 0，我们应该怎么做？由于矩阵中只有一个 0，那么对于每一个 1，离
		它最近的 0 就是那个唯一的 0。如何求出这个距离呢？我们可以想到两
		种做法：
			1、如果 0 在矩阵中的位置是 (i0,j0)，1 在矩阵中的位置是 
				(i1,j1)，那么我们可以直接算出 0 和 1 之间的距离。因为
				我们从 1 到 0 需要在水平方向走 ∣i0−i1∣ 步，竖直方向
				走 ∣j0−j1∣ 步，那么它们之间的距离就为 ∣i0−i1∣+∣j0−j1∣​∣；
			2、我们可以从 0 的位置开始进行 广度优先搜索。广度优先搜索可
			以找到从起点到其余所有点的 最短距离，因此如果我们从 0 开始
			搜索，每次搜索到一个 1，就可以得到 0 到这个 1 的最短距离，
			也就离这个 1 最近的 0 的距离了（因为矩阵中只有一个 0）。
		举个例子，如果我们的矩阵为：
			_ _ _ _
			_ 0 _ _
			_ _ _ _
			_ _ _ _

		其中只有一个 0，剩余的 1 我们用短横线表示。如果我们从 0 开
		始进行广度优先搜索，那么结果依次为：
			_ _ _ _         _ 1 _ _         2 1 2 _         2 1 2 3         2 1 2 3
			_ 0 _ _   ==>   1 0 1 _   ==>   1 0 1 2   ==>   1 0 1 2   ==>   1 0 1 2
			_ _ _ _         _ 1 _ _         2 1 2 _         2 1 2 3         2 1 2 3
			_ _ _ _         _ _ _ _         _ 2 _ _         3 2 3 _         3 2 3 4
		也就是说，在广度优先搜索的每一步中，如果我们从矩阵中的位置 x 搜
		索到了位置 y，并且 y 还没有被搜索过，那么位置 y 离 0 的距离就
		等于位置 x 离 0 的距离加上 1。

		对于「Tree 的 BFS」（典型的「单源 BFS」） 大家都已经轻车熟路了：
        首先把 root 节点入队，再一层一层无脑遍历就行了。
		对于 「图 的 BFS」（「多源 BFS」）做法其实也是一样的，与「Tree的 
		BFS」的区别注意以下两条：
			1、Tree 只有 1 个 root，而图可以有多个源点，所以首先需要把
				多个源点都入队；
			2、Tree 是有向的因此不需要标识是否访问过，而对于无向图来说，
				必须得标志是否访问过！
			并且为了防止某个节点多次入队，需要在其入队之前就将其设置成
			已访问！
		在本题中，我们以 0 作为起始点，把所有的 1 预先置为 -1，然后沿着
		-1 的路径发散，遇到 -1，则把该位置的值改为上一位置的值 +1。
		-1 表示未处理过的点。
	时间复杂度：O(mn)
		m、n 分别是矩阵的行和列，广度优先搜索中每个位置最多只会被加入队
		列一次，因此只需要 O(mn) 的时间复杂度。
	空间复杂度：O(mn)
		m、n 分别是矩阵的行和列，除答案数组外，最坏情况下矩阵里所有元素
		都为 0，全部被加入队列中，此时需要 O(mn) 的空间复杂度。
*/
func updateMatrix(matrix [][]int) [][]int {
    m, n := len(matrix), len(matrix[0])
	// 首先将所有的 0 都入队建立最短路径的超级源点，并且将 1 的位置
	// 设置成 -1，表示该位置是 未被访问过的 1
    queue := make([][]int, 0)
    for i := 0; i < m; i ++ {
        for j := 0; j < n; j ++ {
            if matrix[i][j] == 0 {
                queue = append(queue, []int{i, j})
            } else {
                matrix[i][j] = -1
            }
        }
    }
    for len(queue) > 0 {
        cur := queue[0]
        queue = queue[1:]
        i, j := cur[0], cur[1]
		// 向上下左右发散处理邻居节点，如邻居节点是 -1，则表示这个点是
		// 未被访问过的 1，所以这个点到 0 的距离就可以更新成 
		// matrix[x][y] + 1。
		curVal := matrix[i][j]
        if i - 1 >= 0 && matrix[i - 1][j] == -1 {
			matrix[i-1][j] = curVal+ 1
			queue = append(queue, []int{i-1, j})
		}
		if i + 1 < m && matrix[i + 1][j] == -1 {
			matrix[i+1][j] = curVal + 1
			queue = append(queue, []int{i+1, j})
		}
		if j - 1 >= 0 && matrix[i][j-1] == -1 {
			matrix[i][j-1] = curVal + 1
			queue = append(queue, []int{i, j-1})
		}
		if j + 1 < n && matrix[i][j+1] == -1 {
			matrix[i][j+1] = curVal + 1
			queue = append(queue, []int{i, j+1})
		}
    }
    return matrix
}

/* 
	方法二：动态规划（DP）
	思路：
		由方法一的分析我们可知：
			如果 0 在矩阵中的位置是 (i0,j0)，1 在矩阵中的位置是 
				(i1,j1)，那么我们可以直接算出 0 和 1 之间的距离。因为
				我们从 1 到 0 需要在水平方向走 ∣i0−i1∣ 步，竖直方向
				走 ∣j0−j1∣ 步，那么它们之间的距离就为 ∣i0−i1∣+∣j0−j1∣​∣；
		对于矩阵中的任意一个 1 以及一个 0，我们如何从这个 1 到达 0 并且
		距离最短呢？根据上面的做法，我们可以从 1 开始，先在水平方向移动，
		直到与 0 在同一列，随后再在竖直方向上移动，直到到达 0 的位置。
		这样一来，从一个固定的 1 走到任意一个 0，在距离最短的前提下可
		能有四种方法：
			只有 水平向左移动 和 竖直向上移动；
			只有 水平向左移动 和 竖直向下移动；
			只有 水平向右移动 和 竖直向上移动；
			只有 水平向右移动 和 竖直向下移动。
		这样一来，我们就可以使用动态规划解决这个问题了。我们用 f(i,j)
		表示位置 (i,j) 到最近的 0 的距离。如果我们只能「水平向左移动」
		和「竖直向上移动」，那么我们可以向上移动一步，再移动 f(i−1,j)
		步到达某一个 0，也可以向左移动一步，再移动 f(i,j−1) 步到达某
		一个 0。因此我们可以写出如下的状态转移方程：
					 1+min⁡(f(i−1,j),f(i,j−1)),位置 (i,j) 的元素为 1
			f(i,j)= /
			 		\ 0,位置 (i,j) 的元素为 0
		对于另外三种移动方法，我们也可以写出类似的状态转移方程，得到
		四个 f(i,j) 的值，那么其中最小的值就表示位置 (i,j) 到最近的
		 0 的距离。
	时间复杂度：O(mn)
		m、n 分别是矩阵的行和列，计算 dp 数组的过程中我们需要遍历四次
		矩阵，因此时间复杂度为 O(4mn)=O(mn)
	空间复杂度：O(1)
		里我们只计算额外的空间复杂度。除了答案数组以外，我们只需要常数
		空间存放若干变量。
*/
func updateMatrix(matrix [][]int) [][]int {
	m, n := len(matrix), len(matrix[0])
	dp := make([][]int, m)
	// 初始化 dp 数组，距离都设为很大的数
	for i := 0; i < m; i ++ {
		dp[i] = make([]int, n)
		for j := 0; j < n; j ++ {
			// 如果 (i,j) 元素为 0，那么它到 0 的最短距离为 0
			if matrix[i][j] == 0 {
				dp[i][j] = 0
			} else {
				dp[i][j] = 1 << 31 - 1
			}
		}
	}

	// 只有 水平向左移动 和 竖直向上移动，注意动态规划的计算顺序
	for i := 0; i < m; i ++ {
		for j := 0; j < n; j ++ {
			if (i - 1 >= 0) {
				dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1)
			}
			if (j - 1 >= 0) {
				dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1)
			}
		}
	}
	// 这里注释是优化，中间这两次循环不用重复了
	// // 只有 水平向左移动 和 竖直向下移动，注意动态规划的计算顺序
	// for i := m - 1; i >= 0; i -- {
	// 	for j := 0; j < n; j ++ {
	// 		if (i + 1 < m) {
	// 			dp[i][j] = min(dp[i][j], dp[i + 1][j] + 1)
	// 		}
	// 		if (j - 1 >= 0) {
	// 			dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1)
	// 		}
	// 	}
	// }

	// // 只有 水平向右移动 和 竖直向上移动，注意动态规划的计算顺序
	// for i := 0; i < m; i ++ {
	// 	for j := n - 1; j >= 0; j -- {
	// 		if (i - 1 >= 0) {
	// 			dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1)
	// 		}
	// 		if (j + 1 < n) {
	// 			dp[i][j] = min(dp[i][j], dp[i][j + 1] + 1)
	// 		}
	// 	}
	// }

	// 只有 水平向右移动 和 竖直向下移动，注意动态规划的计算顺序
	for i := m - 1; i >= 0; i -- {
		for j := n - 1; j >= 0; j -- {
			if (i + 1 < m) {
				dp[i][j] = min(dp[i][j], dp[i + 1][j] + 1)
			}
			if (j + 1 < n) {
				dp[i][j] = min(dp[i][j], dp[i][j + 1] + 1)
			}
		}
	}
	return dp
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

/* 
========================== 5、钥匙和房间 =========================
有 N 个房间，开始时你位于 0 号房间。每个房间有不同的号码：
0，1，2，...，N-1，并且房间里可能有一些钥匙能使你进入下一个房间。
在形式上，对于每个房间 i 都有一个钥匙列表 rooms[i]，每个钥匙 
rooms[i][j] 由 [0,1，...，N-1] 中的一个整数表示，其中 
N = rooms.length。 钥匙 rooms[i][j] = v 可以打开编号为 v 的房间。

最初，除 0 号房间外的其余所有房间都被锁住。
【你可以自由地在房间之间来回走动。】
如果能进入每个房间返回 true，否则返回 false。

示例 1：
输入: [[1],[2],[3],[]]
输出: true
解释:  
我们从 0 号房间开始，拿到钥匙 1。
之后我们去 1 号房间，拿到钥匙 2。
然后我们去 2 号房间，拿到钥匙 3。
最后我们去了 3 号房间。
由于我们能够进入每个房间，我们返回 true。

示例 2：
输入：[[1,3],[3,0,1],[2],[0]]
输出：false
解释：我们不能进入 2 号房间。

提示：
    1 <= rooms.length <= 1000
    0 <= rooms[i].length <= 1000
    所有房间中的钥匙数量总计不超过 3000。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/queue-stack/gle1r/
*/
/* 
	方法一：DFS
	思路：
		因为我们可以在房间之间来回走动，所以我们可以优先进入我们已经有钥
		匙的房间，再拿到该房间存储的钥匙进入这些钥匙所对应的房间。
		此时我们可以把房间列表看作是通过对应钥匙相连通的无向图，然后以 0
		号房间作为起点，拿到 0 号房间存储的钥匙，然后向这些钥匙所对应的
		房间发散，发散的过程中记录已经走过的房间。
	时间复杂度：O(m+n)
		m 为房间总数，n 为所有房间中钥匙的总数，每个房间我们只会进入一次，
		但是每个房间的钥匙我们都需要判断该钥匙对应的房间是否已访问。
	空间复杂度：O(m))
		我们需要记录所有房间是否走过。
*/
func canVisitAllRooms(rooms [][]int) bool {
	// 记录已经拿到的钥匙
	visited := make(map[int]bool)
	// 开始时我们是可以进入 0 号房间的
	visited[0] = true

	var DFS func(keys []int) 
	DFS = func(keys []int) {
		for _, v := range keys {
			if !visited[v] {
				// 标记该钥匙对应的房间为已访问
				visited[v] = true
				// 进入该房间获取其存储的钥匙继续处理
				DFS(rooms[v])
			}
		}
	}
	// 传入 0 号房间存储的钥匙
	DFS(rooms[0])
	// 已访问的房间数等于总房间数，说明所有房间都可以进入
	return len(visited) == len(rooms)
}

/* 
	方法二：BFS
	思路：
		我们从 0 号房间开始，循环处理我们能拿到钥匙的每一个房间，处理过程
		中需要记录我们已经访问过大房间。
	时间复杂度：O(m+n)
		m 为房间总数，n 为所有房间中钥匙的总数，每个房间我们只会进入一次，
		但是每个房间的钥匙我们都需要判断该钥匙对应的房间是否已访问。
	空间复杂度：O(m))
		我们需要记录所有房间是否走过。
*/
func canVisitAllRooms(rooms [][]int) bool {
	// 记录已访问的房间
	visited := make(map[int]bool)
	// 标记 0 号房间为已访问
	visited[0] = true
	queue := make([]int, 0)
	// 拿到 0 号房间存储的钥匙
	for _, v := range rooms[0] {
		if !visited[v] {
			queue = append(queue, v)
		}
	}
	// 处理已有钥匙对应的房间
	for len(queue) > 0 {
		// 当前钥匙
		key := queue[0]
		queue = queue[1:]
		// 标记当前钥匙对应的房间为已访问
		visited[key] = true
		// 拿到当前钥匙对应房间所存储的钥匙
		for _, v := range rooms[key] {
			if !visited[v] {
				queue = append(queue, v)
			}
		}
	}
	// 已访问的房间数等于总房间数，说明所有房间都可以进入
	return len(visited) == len(rooms)
}