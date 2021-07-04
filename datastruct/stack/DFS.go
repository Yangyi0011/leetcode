package stack

/* 
	栈在深度优先搜索（DFS）中的应用
	1、二叉树 DFS 模板：
		// 先序遍历
		func preOrder(root *TreeNode) []int {
			if root == nil {
				return []int{}
			}
			result := make([]int, 0)
			stack := make([]*TreeNode, 0)
			for root != nil || len(stack) > 0 {
				for root != nil {
					// Push
					stack = append(stack, root)
					result = append(result, root.Val)
					root = root.Left
				}
				// Pop
				node := stack[len(stack) - 1]
				stack = stack[:len(stack) - 1]
				root = node.Rgiht
			}
			return result
		}
	2、搜索 DFS 模板：
		// cur：当前节点
		// target：搜索的目标节点
		// visited：已访问节点列表
		func DFS(cur *Node, target *Node, visited map[*Node]bool) bool {
			if cur == target {
				return true
			}
			// 获取所有当前节点可达的邻居节点
			var neighbor []*Node
			// 遍历邻居节点
			for _, next := range neighbor {
				// 如果该邻居节点还未被访问，则处理它
				if _, ok := visited[next]; !ok {
					visited[next] = true
					return DFS(next, target, visited)
				}
			}
			return false
		}

	递归解决方案的优点是它更容易实现。 但是，存在一个很大的缺点：如果递归
	的深度太高，你将遭受堆栈溢出。 在这种情况下，您可能会希望使用 BFS，
	或使用显式栈实现 DFS。
	这里我们提供了一个使用显式栈的模板：
		// Return true if there is a path from cur to target.
		boolean DFS(int root, int target) {
			Set<Node> visited;
			Stack<Node> s;
			add root to s;
			while (s is not empty) {
				Node cur = the top element in s;
				return true if cur is target;
				for (Node next : the neighbors of cur) {
					if (next is not in visited) {
						add next to s;
						add next to visited;
					}
				}
				remove cur from s;
			}
			return false;
		}
*/
/* 
========================== 1、岛屿数量 =========================
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛
屿的数量。
岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地
连接形成。
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
链接：https://leetcode-cn.com/leetbook/read/queue-stack/gpfm5/
*/
/* 
	方法一：DFS
	思路：
		把网格看成是通过 '1' 进行连接的无向图，我们以每一个未访问过
		的 '1' 为起点，往 上下左右 四个方向对网格进行 DFS 搜索，则
		可以 DFS 的次数即为岛屿的数量。
		注意：需要记录已经访问过的节点，以避免重复访问和死循环。
		在本题中，我们可以通过把已访问的 '1' 变为 '0' 的方式来标记
		已访问状态。
	时间复杂度：O(mn)
		其中 m 和 n 分别为行数和列数。
	空间复杂度：O(mn)
		在最坏情况下，整个网格均为陆地，深度优先搜索的深度达到 mn。
*/
func numIslands(grid [][]byte) int {
	m, n := len(grid), len(grid[0])
	var DFS func(i, j int)
	DFS = func(i, j int) {
		if i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == '0' {
			return
		}
		// 标记为已访问
		grid[i][j] = '0'
		DFS(i - 1, j)
		DFS(i + 1, j)
		DFS(i, j - 1)
		DFS(i, j + 1)
	}
	cnt := 0
	for i := 0; i < m; i ++ {
		for j := 0; j < n; j ++ {
			if grid[i][j] == '1' {
				DFS(i, j)
				cnt ++
			}
		}
	}
	return cnt
}
/* 
	方法二：BFS
	思路：
		对于 DFS 的题目，我们也都可以用 BFS 来处理，需要借助于队列来实现。
	时间复杂度：O(mn)
		其中 m 和 n 分别为行数和列数。
	空间复杂度：O(min⁡(m,n))
		在最坏情况下，整个网格均为陆地，队列的大小可以达到 min⁡(m,n)。
*/
func numIslands(grid [][]byte) int {
	m, n := len(grid), len(grid[0])
	var BFS func(i, j int)
	BFS = func(i, j int) {
		// 存入 i、j 的下标
		queue := [][]int{[]int{i, j}}
		for len(queue) > 0 {
			// 取出 i、j 的下标
			cur := queue[0]
			queue = queue[1:]
			row := cur[0]
			col := cur[1]
			if row >= 0 && row < m && col >= 0 && col < n && grid[row][col] == '1' {
				// 标记为已访问
				grid[row][col] = '0'
				// 向上下左右扩散
				queue = append(queue, []int{row-1, col})
				queue = append(queue, []int{row+1, col})
				queue = append(queue, []int{row, col-1})
				queue = append(queue, []int{row, col+1})
			}
		}
	}
	cnt := 0
	for i := 0; i < m; i ++ {
		for j := 0; j < n; j ++ {
			if grid[i][j] == '1' {
				BFS(i, j)
				cnt ++
			}
		}
	}
	return cnt
}

/* 
========================== 2、克隆图 =========================
给你无向 连通 图中一个节点的引用，请你返回该图的 深拷贝（克隆）。
图中的每个节点都包含它的值 val（int） 和其邻居的列表（list[Node]）。
class Node {
    public int val;
    public List<Node> neighbors;
}

测试用例格式：
简单起见，每个节点的值都和它的索引相同。例如，第一个节点值为 1（val = 1），
第二个节点值为 2（val = 2），以此类推。该图在测试用例中使用邻接列表表示。
邻接列表 是用于表示有限图的无序列表的集合。每个列表都描述了图中节点的邻居集。
给定节点将始终是图中的第一个节点（值为 1）。你必须将 给定节点的拷贝 作为对
克隆图的引用返回。

示例 1：
输入：adjList = [[2,4],[1,3],[2,4],[1,3]]
输出：[[2,4],[1,3],[2,4],[1,3]]
解释：
图中有 4 个节点。
节点 1 的值是 1，它有两个邻居：节点 2 和 4 。
节点 2 的值是 2，它有两个邻居：节点 1 和 3 。
节点 3 的值是 3，它有两个邻居：节点 2 和 4 。
节点 4 的值是 4，它有两个邻居：节点 1 和 3 。

示例 2：
输入：adjList = [[]]
输出：[[]]
解释：输入包含一个空列表。该图仅仅只有一个值为 1 的节点，它没有任何邻居。

示例 3：
输入：adjList = []
输出：[]
解释：这个图是空的，它不含任何节点。

示例 4：
输入：adjList = [[2],[1]]
输出：[[2],[1]]

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/queue-stack/gmcr6/
*/
/* 
	方法一：DFS
	思路：
		DFS自底向上克隆节点及其邻居列表，当每一个节点的邻居列表都克隆完成
		后，再返回克隆节点本身。注意克隆过程中需要记录已处理的节点，以避免
		重复克隆和死循环。
	时间复杂度：O(n)
		n 是无向图的节点个数，对于每一个节点我们至多只克隆一次。
	空间复杂度：O(n)
		存储克隆节点和原节点的哈希表需要 O(n)) 的空间，递归调用栈需
		要 O(H) 的空间，其中 H 是图的深度，经过放缩可以得到 O(H)=O(n)，
		因此总体空间复杂度为 O(n)。
*/
func cloneGraph(node *Node) *Node {
    if node == nil {
		return nil
	}
	// 记录已访问节点，key：旧节点，value：新节点
	visited := make(map[*Node]*Node)
	var DFS func(node *Node) *Node
	DFS = func(node *Node) * Node {
		// 如果该节点已经被访问过了，则直接从哈希表中取出对应的克隆节点返回
        if _, ok := visited[node]; ok {
            return visited[node]
        }
		// 克隆节点，注意到为了深拷贝此时我们不会克隆它的邻居的列表
        cloneNode := &Node{node.Val, []*Node{}}
		// 标记为已处理
		visited[node] = cloneNode
		// 克隆邻居列表
		for _, v := range node.Neighbors {
			cloneNode.Neighbors = append(cloneNode.Neighbors, DFS(v))
		}
		// 返回克隆节点
		return cloneNode
	}
	return DFS(node)
}

/* 
	方法二：BFS
	思路：
		BFS 逐一克隆遇到的每一个节点，并把已克隆的节点标记为已处理。
	时间复杂度：O(n)
		n 是无向图的节点个数，对每一个节点我们只需要克隆一次。
	空间复杂度：O(n)
		哈希表使用 O(n)) 的空间。广度优先搜索中的队列在最坏情况下会达
		到 O(n) 的空间复杂度，因此总体空间复杂度为 O(n)。
*/
func cloneGraph(node *Node) *Node {
	if node == nil {
		return nil
	}
	// 记录已处理节点，key：旧节点，value：新节点
	visited := map[*Node]*Node{}
	// 将题目给定的节点添加到队列
	queue := []*Node{node}
	// 克隆第一个节点并存储到哈希表中
	// 这样提前处理的好处是在 BFS 中只需要从邻居列表开始处理就行了
	visited[node] = &Node{node.Val, []*Node{}}
	// 广度优先搜索
	for len(queue) > 0 {
		// 取出队列的头节点，元素在入队前都是已处理过了的
		n := queue[0]
		queue = queue[1:]
		// 遍历该节点的邻居
		for _, neighbor := range n.Neighbors {
			if _, ok := visited[neighbor]; !ok {
				// 如果没有被处理过，就克隆并存储在哈希表中
				visited[neighbor] = &Node{neighbor.Val, []*Node{}}
				// 将邻居节点加入队列中
				queue = append(queue, neighbor)
			}
			// 更新当前已克隆节点的邻居列表
			visited[n].Neighbors = append(visited[n].Neighbors, visited[neighbor])
		}
	}
	return visited[node]
}

/* 
========================== 3、目标和 =========================
给定一个非负整数数组，a1, a2, ..., an, 和一个目标数，S。现在你有两个符
号 + 和 -。对于数组中的任意一个整数，你都可以从 + 或 -中选择一个符号添
加在前面。
返回可以使最终数组和为目标数 S 的所有添加符号的方法数。

示例：
输入：nums: [1, 1, 1, 1, 1], S: 3
输出：5
解释：
-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3

一共有5种方法让最终目标和为3。

提示：
    数组非空，且长度不会超过 20 。
    初始的数组的和不会超过 1000 。
    保证返回的最终结果能被 32 位整数存下

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/queue-stack/ga4o2/
*/
/* 
	方法一：递归
	思路：
		我们可以使用递归，枚举出所有可能的情况。具体地，当我们处理到第 
		i 个数时，我们可以将它添加 + 或 -，递归地搜索这两种情况。当我们
		处理完所有的 N 个数时，我们计算出所有数的和，并判断是否等于 S，
		等于时，满足条件的次数就 +1。
	时间复杂度：O(2^n)
		其中 n 数组 nums 的长度
	空间复杂度：O(n)
		为递归使用的栈空间大小
*/
func findTargetSumWays(nums []int, S int) int {
	n := len(nums)
	count := 0
	var DFS func(i, sum int)
	DFS = func(i, sum int) {
		if i == n {
			// 处理完 n 个数了
			if sum == S {
				count ++
			}
		} else {
			DFS(i + 1, sum + nums[i])
			DFS(i + 1, sum - nums[i])
		}
	}
	DFS(0, 0)
	return count
}
/* 
	方法二：DP
	思路：
		这道题也是一个常见的背包问题，我们可以用类似求解背包问题的方法来
		求出可能的方法数。
		我们用 dp[i][j] 表示用数组中的前 i 个元素，组成和为 j 的方案数。
		考虑第 i 个数 nums[i]，它可以被添加 + 或 -，因此状态转移方程如下：
			dp[i][j] = dp[i - 1][j - nums[i]] + dp[i - 1][j + nums[i]]
		也可以写成递推的形式：
			dp[i][j + nums[i]] += dp[i - 1][j]
			dp[i][j - nums[i]] += dp[i - 1][j]
		由于数组中所有数的和不超过 1000，那么 j 的最小值可以达到 -1000。
		在很多语言中，是不允许数组的下标为负数的，因此我们需要给 dp[i][j]
		的第二维预先增加 1000，即：
			dp[i][j + nums[i] + 1000] += dp[i - 1][j + 1000]
			dp[i][j - nums[i] + 1000] += dp[i - 1][j + 1000]
	时间复杂度：O(n∗sum)
		其中 n 是数组 nums 的长度。
	空间复杂度：O(n∗sum)
*/
func findTargetSumWays(nums []int, S int) int {
	n := len(nums)
	dp := make([][]int, n)
	for i := 0; i < n; i ++ {
		dp[i] = make([]int, 2001)
	}
	// 初始状态
	dp[0][0 + nums[0] + 1000] += 1
	dp[0][0 - nums[0] + 1000] += 1
	for i := 1; i < n; i ++ {
		for sum := -1000; sum <= 1000; sum ++ {
			// 要的是最多方案数，所以只加方案数为 正 的
			if dp[i - 1][sum + 1000] > 0 {
				dp[i][sum + nums[i] + 1000] += dp[i - 1][sum + 1000]
				dp[i][sum - nums[i] + 1000] += dp[i - 1][sum + 1000]
			}
		}
	}
	if S > 1000 {
		return 0
	}
	return dp[n - 1][S + 1000]
}

/* 
	方法三：动态规划 + 空间优化
	思路：
		我们发现，方法二中动态规划的状态转移方程中，dp[i][...] 只和 
		dp[i - 1][...] 有关，因此我们可以优化动态规划的空间复杂度，
		只需要使用两个一维数组即可。
	时间复杂度：O(n*sum)
		其中 n 是数组 nums 的长度
	空间复杂度：O(sum))
*/
func findTargetSumWays(nums []int, S int) int {
	dp := make([]int, 2001)
	dp[0 + nums[0] + 1000] += 1
	dp[0 - nums[0] + 1000] += 1
	for i := 1; i < len(nums); i ++ {
		next := make([]int, 2001)
		for sum := -1000; sum <= 1000; sum ++ {
			if dp[sum + 1000] > 0 {
				next[sum + nums[i] + 1000] += dp[sum + 1000]
				next[sum - nums[i] + 1000] += dp[sum + 1000]
			}
		}
		dp = next
	}
	if S > 1000 {
		return 0
	}
	return dp[S + 1000]
}