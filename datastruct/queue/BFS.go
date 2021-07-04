package queue

/* 
	队列与广度优先算法
		队列通常用于广度优先算法（BFS）中，一般是用来遍历树的层级结构或者是从
		图中找出从根结点到目标结点的最短路径。
				1:		A
					 /  |  \
				2:	D   C   B
					|  /  \ |
				3:	|  F    E
					| /
					G

		1. 结点的处理顺序是什么？
			在第一轮中，我们处理根结点。在第二轮中，我们处理根结点旁边的结点；
			在第三轮中，我们处理距根结点两步的结点；等等等等。
			与树的层序遍历类似，越是接近根结点的结点将越早地遍历。
			如果在第 k 轮中将结点 X 添加到队列中，则根结点与 X 之间的最短路径
			的长度恰好是 k。也就是说，第一次找到目标结点时，你已经处于最短路径
			中。

		2. 队列的入队和出队顺序是什么？
			如上面的动画所示，我们首先将根结点排入队列。然后在每一轮中，我们
			逐个处理已经在队列中的结点，并将所有邻居添加到队列中。值得注意的
			是，新添加的节点不会立即遍历，而是在下一轮中处理。
			结点的处理顺序与它们添加到队列的顺序是完全相同的顺序，即先进先出
			（FIFO）。这就是我们在 BFS 中使用队列的原因。
*/
// 广度优先搜索（BFS）模板一：
// 在树中寻找目标节点的最短路径距离，找不到返回 -1
func BFS(root *TreeNode, target *TreeNode) int {
	if root == nil {
		return -1
	}
	queue := []*TreeNode{root}
	// 走到目标节点需要的最小步数
	step := 0
	for len(queue) > 0 {
		step ++
		n := len(queue)
		for _, node := range queue {
			// 找到目标节点
			if node == target {
				return step
			}
			// 把相邻节点入队
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		// 移除已经处理的节点
		queue = queue[n:]
	}
	// 找不到
	return -1
}

// BFS模板二：
// 如果是处理图类题目，可能会出现重复访问某个节点的情况
// 我们需要用哈希表记录哪些节点已经访问过，以避免死循环
func BFS2(root *Node, target *Node) int {
	if root == nil {
		return -1
	}
	// 记录已经访问过的节点
	visited := make(map[*Node]bool)
	queue := []*Node{root}
	step := 0
	for len(queue) > 0 {
		step ++
		n := len(queue)
		for _, node := range queue {
			// 找到目标节点
			if node == target {
				return step
			}
			// 没有访问过的节点才需要处理
			if _, ok := visited[node]; !ok {
				// 把 node 的相邻节点入队，Neighbors 表示 node 的相邻节点集合
				for _, v := range node.Neighbors {
					queue = append(queue, v)
				}
				// 标记 node 为已访问
				visited[node] = true
			}
		}
		// 移除以处理节点
		queue := queue[n:]
	}
	// 找不到
	return -1
}

/* 
========================== 1、岛屿数量 ==========================
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
此外，你可以假设该网格的四条边均被水包围。
 
示例 1：
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1

示例 2：
输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3

提示：
    m == grid.length
    n == grid[i].length
    1 <= m, n <= 300
    grid[i][j] 的值为 '0' 或 '1'

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/queue-stack/kbcqv/
*/
/* 
	方法一：深度优先搜索（DFS）
	思路：
		我们可以将二维网格看成一个无向图，竖直或水平相邻的 1 之间有边相连。
		为了求出岛屿的数量，我们可以扫描整个二维网格。如果一个位置为 1，则
		以其为起始节点开始进行深度优先搜索。

		即把每一个'1'（陆地）作为起始点，沿着'1'出发做深度优先搜索，遇
		到'1'（陆地）就继续前行，否则停止。按此思路，我们可以沿着陆地
		向上、下、左、右四个方向进行搜索，当每一个方向都走不通时，说明此块
		陆地是一个岛屿，即岛屿数量+1。
		注意：在（发散）搜索过程中，我们需要记录已经访问过的'1'（陆地），以
			避免重复搜索和死循环。此处我们可以这样做：我们在搜索过程中每走过
			一个'1'(陆地)就把它变为'0'(水)，这样就不会重复搜索了。
	时间复杂度：O(m*n)
		m、n 分别是网格的行数和列数。
	空间复杂度：O(m*n)
		在最坏情况下，整个网格均为陆地，深度优先搜索的深度达到 m*n。
*/
func numIslands(grid [][]byte) int {
	m, n := len(grid), len(grid[0])
	landCount := 0
	var DFS func(i, j int) 
	DFS = func(i, j int) {
		// 边界返回条件
		if i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == '0'{
			return
		}
		// 把走过的位置置为 '0'
		grid[i][j] = '0'
		// 向上、下、左、右发散
		DFS(i - 1, j)
		DFS(i + 1, j)
		DFS(i, j - 1)
		DFS(i, j + 1)
	}
	for i := 0; i < m; i ++ {
		for j := 0; j < n; j ++ {
			if grid[i][j] == '1' {
				DFS(i, j)
				landCount ++
			}
		}
	}
	return landCount
}

/* 
	方法二：广度优先搜索（BFS）
	思路：
		同样地，我们也可以使用广度优先搜索代替深度优先搜索。
		为了求出岛屿的数量，我们可以扫描整个二维网格。如果一个位置为 1，则将
		其加入队列，开始进行广度优先搜索。在广度优先搜索的过程中，每个搜索
		到的 1 都会被重新标记为 0。直到队列为空，搜索结束。
		最终岛屿的数量就是我们进行广度优先搜索的次数。
	时间复杂度：O(m*n)
		m、n 分别是网格的行数和列数。
	空间复杂度：O(min⁡(m,n))
		在最坏情况下，整个网格均为陆地，队列的大小可以达到 min⁡(m,n)
*/
func numIslands(grid [][]byte) int {
	m, n := len(grid), len(grid[0])
	landCount := 0
	for i := 0; i < m; i ++ {
		for j := 0; j < n; j ++ {
			if grid[i][j] == '1' {
				grid[i][j] = '0'
				landCount ++
				// 把 i、j 的值用 i * n + j 计算放入队列中
				queue := []int{i * n + j}
				for len(queue) > 0 {
					// 取出 i、j
					id := queue[0]
					queue = queue[1:]
					row := id / n
					col := id % n
					// 向上、下、左、右四个方向添加 grid[i][j] 的相邻节点到队列
					if row - 1 >= 0 && grid[row-1][col] == '1' {
						grid[row-1][col] = '0'
						queue = append(queue, (row - 1) * n + col)
					}
					if row + 1 < m && grid[row+1][col] == '1' {
						grid[row+1][col] = '0'
						queue = append(queue, (row + 1) * n + col)
					}
					if col - 1 >= 0 && grid[row][col-1] == '1' {
						grid[row][col-1] = '0'
						queue = append(queue, row * n + col - 1)
					}
					if col + 1 < n && grid[row][col+1] == '1' {
						grid[row][col+1] = '0'
						queue = append(queue, row * n + col + 1)
					}
				}
			}
		}
	}
	return landCount
}

/* 
========================== 2、打开转盘锁 ==========================
你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： '0', '1', '2', 
'3', '4', '5', '6', '7', '8', '9' 。每个拨轮可以自由旋转：例如把 '9' 变
为  '0'，'0' 变为 '9' 。每次旋转都只能旋转一个拨轮的一位数字。
锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。
列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，
这个锁将会被永久锁定，无法再被旋转。
字符串 target 代表可以解锁的数字，你需要给出最小的旋转次数，如果无论如何不能
解锁，返回 -1。

示例 1:
输入：deadends = ["0201","0101","0102","1212","2002"], target = "0202"
输出：6
解释：
可能的移动序列为 "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202"。
注意 "0000" -> "0001" -> "0002" -> "0102" -> "0202" 这样的序列是不能解锁的，
因为当拨动到 "0102" 时这个锁就会被锁定。

示例 2:
输入: deadends = ["8888"], target = "0009"
输出：1
解释：
把最后一位反向旋转一次即可 "0000" -> "0009"。

示例 3:
输入: deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"], target = "8888"
输出：-1
解释：
无法旋转到目标数字且不被锁定。

示例 4:
输入: deadends = ["0000"], target = "8888"
输出：-1

提示：
    死亡列表 deadends 的长度范围为 [1, 500]。
    目标数字 target 不会在 deadends 之中。
    每个 deadends 和 target 中的字符串的数字会在 10,000 个可能的情况 '0000' 到 '9999' 中产生。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/queue-stack/kj48j/
*/
/* 
	方法一：广度优先搜索（BFS）
	思路：
		我们可以将 0000 到 9999 这 10000 种状态看成图上的 10000 个节点，两个
		节点之间存在一条边，当且仅当这两个节点对应的状态只有 1 位不同，且不同
		的那位相差 1（包括 0 和 9 也相差 1 的情况），并且这两个节点均不在数
		组 deadends 中。那么最终的答案即为 0000 到 target 的最短路径。

		我们用广度优先搜索来找到最短路径，从 0000 开始搜索。对于每一个状态，它
		可以扩展到最多 8 个状态，即将它的第 i = 0, 1, 2, 3 位增加 1 或减少 1，
		将这些状态中没有搜索过并且不在 deadends 中的状态全部加入到队列中，并继
		续进行搜索。注意 0000 本身有可能也在 deadends 中。
	缺点：使用了 map[string]int来记录步数，int是32bit或64bit，而实际上 
		visited 只需要 bool 型，最极端下还可以用interface{}使得内存占用更少。
		至于记录步数，可以用一个变量来存储。还有，'0'-'9'是可以直接用byte存储的，
		不需要rune
*/
func openLock(deadends []string, target string) int {
	// 记录死锁状态
	dead := make(map[string]bool)
	for _, v := range deadends {
		dead[v] = true
	}
	// 起点即为终点
	if target == "0000" {
		return 0
	}
	// 直接死锁
	if dead["0000"] {
		return -1
	}
	// BFS
	queue := []string{"0000"}
	// 已访问过的集合，并记录到达该状态需要走的步数
	visited := make(map[string]int)
	visited["0000"] = 0

	for len(queue) > 0 {
		// 取出当前的锁状态（无向图节点）
		cur := queue[0]
		queue = queue[1:]
		// 找到目标
		if cur == target {
			return visited[cur]
		}

		// 转为 slice
		curSlice := []rune(cur)
		// 获取当前节点的所有相邻节点（8个，四个锁轮向上或向下滚动 1 位）
		var nexts []string
		for i := 0; i < 4; i ++ {
			// 备份源字符
			origin := curSlice[i]
			// 正向转动转盘，对单个字符做状态变化
			// '0'~'9'的字符减去'0' 变为整型，来和1作加减，外边再 + '0'又转为字符
			curSlice[i] = (curSlice[i] - '0' + 1) % 10 + '0'
			// 把变化后的节点加入邻居节点集合
			nexts = append(nexts, string(curSlice))
			// 以变化的字符恢复到原始状态
			curSlice[i] = origin

			// 反向转动转盘
			curSlice[i] = (curSlice[i] - '0' + 9) % 10 + '0'
			nexts = append(nexts, string(curSlice))
			curSlice[i] = origin
		}

		// 遍历下一步的所有可能状态
		for _, next := range nexts {
			if _, ok := visited[next]; !ok && !dead[next] {
				queue = append(queue, next)
				// 下一节点所需的步数是当前节点所需步数 + 1
				visited[next] = visited[cur] + 1
			}
		}
	}
	return -1
}

/* 
	方法二：单向BFS-优化
	思路：
		前面已经分析了上面写法的一些小问题，现在做一些改动，至于这个步数，
		还可以放到 queue 来实现，将状态的 string 和步数捆在一起作为结构体。
		弄一个结构体队列
*/
func openLock(deadends []string, target string) int {
	dead := make(map[string]bool, 0)
	for _, v := range deadends {
		dead[v] = true
	}
	if dead["0000"] {
		return -1
	}
	if "0000" == target {
		return 0
	}
	// BFS
	// 构造处理字符串队列
	queue := []string{"0000"}
	// 已访问过的集合。由于总共只有一万个状态点，所以步数不可能需要更多，所以uint16足以表示
	visited := make(map[string]int16)
	visited["0000"] = 0

	var cur string		// 当前节点（字符串）
	var curSlice []byte	// 当前节点对应的 Byte 数组
	var nexts [8]string	// 当前节点的邻居节点
	var origin byte		// 当前节点状态变化的原始状态
	for len(queue) > 0 {
		cur = queue[0]
		queue = queue[1:]
		if cur == target {
			return int(visited[cur])
		}

		curSlice = []byte(cur)
		// 获取当节点的所有可能的状态变化（相邻节点）
		for i := 0; i < 4; i ++ {
			// 记录原始状态
			origin = curSlice[i]

			// 正向旋转转盘
			curSlice[i] = (curSlice[i] - '0' + 1) % 10 + '0'
			// 记录相邻节点
			nexts[2*i] = string(curSlice)
			// 状态恢复
			curSlice[i] = origin

			// 反向旋转转盘
			curSlice[i] = (curSlice[i] - '0' + 9) % 10 + '0'
			nexts[2*i+1] = string(curSlice)
			curSlice[i] = origin
		}

		// 处理相邻节点
		for _, next := range nexts {
			if _, ok := visited[next]; !ok && !dead[next] {
				// 相邻节点入队
				queue = append(queue, next)
				visited[next] = visited[cur] + 1
			}
		}
	}
	return -1
}

/* 
========================== 3、完全平方数 ==========================
给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和
等于 n。你需要让组成和的完全平方数的个数最少。

给你一个整数 n ，返回和为 n 的完全平方数的 最少数量 。

完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数
自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。

示例 1：
输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4

示例 2：
输入：n = 13
输出：2
解释：13 = 4 + 9

提示：
    1 <= n <= 10^4

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/queue-stack/kfgtt/
*/
/* 
	方法一：暴力枚举法 [超出时间限制]
	思路：
		这个问题要求我们找出由完全平方数组合成给定数字的最小个数。我们将问题
		重新表述成：
			给定一个完全平方数列表和正整数 n，求出完全平方数组合成 n 的组合，
			要求组合中的解拥有完全平方数的最小个数。
			注：可以重复使用列表中的完全平方数。
		从上面对这个问题的叙述来看，它似乎是一个组合问题，对于这个问题，一个直
		观的解决方案是使用暴力枚举法，我们枚举所有可能的组合，并找到完全平方数
		的个数最小的一个。
		我们可以用下面的公式来表述这个问题：
			numSquares(n)=min⁡(numSquares(n-k) + 1) ∀ k∈square
*/
func numSquares(n int) int {
	// 生成完全平方序列
	hash := make(map[int]interface{})
	for i := 1; i < int(math.Floor(math.Sqrt(float64(n)))) + 1; i ++ {
		hash[i*i] = nil
	}
	var minSquareCnt func(k int) int
	minSquareCnt = func(k int) int {
		// 找到符合条件的平方数，次数返回 1
		if _, ok := hash[k]; ok {
			return 1
		}
		// 默认最小次数为 10000
		minCnt := 10000
		// 遍历平方数序列
		for square, _ := range hash {
			if k < square {
				break
			}
			// 递归查找
			cnt := minSquareCnt(k - square) + 1
			minCnt = min(minCnt, cnt)
		}
		return minCnt
	}
	return minSquareCnt(n)
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

/* 
	方法二：动态规划
	思路：
		使用暴力枚举法会超出时间限制的原因很简单，因为我们重复的计算了中间解。
		我们以前的公式仍然是有效的。我们只需要一个更好的方法实现这个公式。
			numSquares(n)=min⁡(numSquares(n-k) + 1)∀k∈square
		通过上面公式我们知道，
		想要得到 n 所需的完全平方数的最小值，就需要先求得 n - k 所需完全平方
		数的最小值，类似于斐波那契数列那样，如果只是单纯的递归，就会做很多重复
		计算，所以我们需要记录这些已经处理好的中间结果，避免重复计算。

		我们用 dp[i] 来表示 i 所需的完全平方数的最小个数，i > 0，则 
			dp[i] = dp[i-k] + 1, ∀k ∈ square
		即对任意 k 属于完全平方数序列时，dp[i] = dp[i-k] + 1
		初始状态：
			dp[0] = 0
		中间状态：
			dp[i] = dp[i-k] + 1, ∀k ∈ square
		终止状态：
			dp[n]
			在循环结束时，我们返回数组中的最后一个元素作为解决方案的结果。
	时间复杂度：O(n⋅sqrt{n})
		在主步骤中，我们有一个嵌套循环，其中外部循环是 n 次迭代，而内部循环最多
		需要 sqrt{n}迭代。
	空间复杂度：O(n)
		使用了一个一维数组 dp。
*/
func numSquares(n int) int {
	// dp[i] 表示到达目标数 i 所需的最少完全平方数的个数
	// 因为有 dp[0]，所以 dp 数组是 n + 1
	dp := make([]int, n+1)
	// 初始化 dp
	for i := 1; i < n + 1; i ++ {
		dp[i] = 10000
	}
	dp[0] = 0

	// 生成完全平方序列
	hash := make(map[int]interface{})
	for i := 1; i < int(math.Floor(math.Sqrt(float64(n)))) + 1; i ++ {
		hash[i*i] = nil
	}
	// 从 1 遍历到 n + 1（dp[n]为目标）
	for i := 1; i < n + 1; i ++ {
		// 遍历完全平方序列
		for square, _ := range hash {
			if i < square {
				continue
			}
			// 寻找 dp[i] 的最小值
			dp[i] = min(dp[i], dp[i-square] + 1)
		}
	}
	return dp[n]
}

/* 
	方法三：贪心枚举
	思路：
		递归解决方法为我们理解问题提供了简洁直观的方法。我们仍然可以用递归解决
		这个问题。为了改进上述暴力枚举解决方案，我们可以在递归中加入贪心。我们
		可以将枚举重新格式化如下：
			从一个数字到多个数字的组合开始，一旦我们找到一个可以组合成给定数
			字 n 的组合，那么我们可以说我们找到了最小的组合，因为我们贪心的从
			小到大的枚举组合。

		为了更好的解释，我们首先定义一个名为 is_divided_by(n, count) 的函数，
		该函数返回一个布尔值，表示数字 n 是否可以被一个数字 count 组合，而不是
		像前面函数 numSquares(n) 返回组合的确切大小。
			numSquares(n) =    arg min       (is_divided_by(n,count))
							count∈[1,2,...n]	
		与递归函数 numSquare(n) 不同，is_divided_by(n, count) 的递归过程可
		以归结为底部情况（即 count==1）更快。
		算法：
			1、首先，我们准备一个小于给定数字 n 的完全平方数列表（称为 
				square_nums）。
			2、在主循环中，将组合的大小（称为 count）从 1 迭代到 n，我们检查
				数字 n 是否可以除以组合的和，即 is_divided_by(n, count)。
			3、函数 is_divided_by(n, count) 可以用递归的形式实现，汝上面所说。
			4、在最下面的例子中，我们有 count==1，我们只需检查数字 n 是否本身
				是一个完全平方数。可以在 square_nums 中检查，即 n∈square_numsn。
				如果 square_nums 使用的是集合数据结构，我们可以获得比 
				n == int(sqrt(n)) ^ 2 更快的运行时间。
	时间复杂度：O(n^(h/2)​)，
		其中 h 是可能发生的最大递归次数。你可能会注意到，上面的公式实际
		上类似于计算完整 N 元数种结点数的公式。事实上，算法种的递归调用
		轨迹形成一个 N 元树，其中 N 是 square_nums 种的完全平方数个数。
		即，在最坏的情况下，我们可能要遍历整棵树才能找到最终解。
	空间复杂度：O(sqrt{n})
		我们存储了一个列表 square_nums，我们还需要额外的空间用于递归调
		用堆栈。但正如我们所了解的那样，调用轨迹的大小不会超过 4。
*/
func numSquares(n int) int {
	hash := make(map[int]interface{}, 0)
	// n 是否可以被 count 个数字组合
	var is_divided_by func(n, count int) bool 
	is_divided_by = func(n, count int) bool{
		// count = 1 时，需要 n 在平方数序列中
		if count == 1 {
			_, ok := hash[n]
			return ok
		}
		// 每有一个平方数（square）符合，n 就减去这个平方数，
		// 继续找下一个符合条件的平方数，平方数序列中的数可以
		// 重复使用
		for square, _ := range hash {
			if is_divided_by(n - square, count - 1) {
				return true
			}
		}
		return false
	}

	// 初始化平方数序列
	for i := 1; i * i <= n; i ++ {
		hash[i*i] = nil
	}
	// 贪心算法，最好情况下，一个平方数就可以表示 n
	count := 1
	// 需要的平方数的个数 count 不断递增，最多为 n，即最多需要 n 个 1 相加
	for ; count <= n; count ++ {
		if is_divided_by(n, count) {
			return count
		}
	}
	return count
}

/* 
	方法四：数学运算
	思路：
		1、拉格朗日四平方定理：每个自然数都可以表示为四个整数平方和：
			p = a^2 + b^2 + c^2 + d^2
			如：3、31可以如下表示
				3 = 1^2 + 1^2 + 1^2 + 0^2
				31 = 5^2 + 2^2 + 1^2 + 1^2
			拉格朗日四平方定理设置了问题结果的上界，即如果数 n 不能分
			解为较少的完全平方数，则至少可以分解为 4个完全平方数之和，
			即 numSquares(n)≤4。
			但是拉格朗日四平方定理并没有直接告诉我们用最小平方数来分解自然数。
		2、Adrien Marie Legendre用他的三平方定理完成了四平方定理，证明
			了正整数可以表示为三个平方和的一个特殊条件：
				n≠4k(8m+7)  ⟺  n=a^2+b^2+c^2
			其中 k 和 m 是整数。
			Adrien-Marie-Legendre 的三平方定理给了我们一个充分必要的条
			件来检验这个数是否只能分解成 4 个平方。
		3、如果这个数满足三平方定理的条件，则可以分解成三个完全平方数。但
			我们不知道的是，如果这个数可以分解成更少的完全平方数，即一个或
			两个完全平方数。
			所以在我们把这个数视为底部情况（三平方定理）之前，还有两种情况
			需要检查，即：
				（1）如果数字本身是一个完全平方数，这很容易检查，
					例如 n == int(sqrt(n)) ^ 2。
				（2）如果这个数可以分解成两个完全平方数和。不幸的是，
					没有任何数学定理可以帮助我们检查这个情况。我们需要
					使用枚举方法。
		按照上面的例子来实现解决方案：
			1、首先，我们检查数字 n 的形式是否为 n=4^k(8m+7)，如果是，
				则直接返回 4。
			2、否则，我们进一步检查这个数本身是否是一个完全平方数，或者这个
				数是否可以分解为两个完全平方数和。
			3、在底部的情况下，这个数可以分解为 3 个平方和，但我们也可以根
				据四平方定理，通过加零，把它分解为 4 个平方。但是我们被要求
				找出最小的平方数。
	时间复杂度：O(sqrt{n})
		在主循环中，我们检查数字是否可以分解为两个平方和，这需要 
		O(sqrt{n}) 个迭代。在其他情况下，我们会在常数时间内进行检查。
	空间复杂度：O(1)
		该算法消耗一个常量空间。

*/
func numSquares(n int) int {
	var isSquare func(n int) bool
	isSquare = func(n int) bool {
		sq := int(math.Floor(math.Sqrt(float64(n))))
		return n == sq * sq
	}
	for n % 4 == 0 {
		n /= 4
	}
	if (n % 8 == 7) {
		return 4
	}
	if isSquare(n) {
		return 1
	}
	for i := 1; i * i <= n; i ++ {
		if isSquare(n - i*i) {
			return 2
		}
	}
	return 3
}