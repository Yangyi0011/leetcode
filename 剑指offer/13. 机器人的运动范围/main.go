package main

import "fmt"

/*
============== 剑指 Offer 13. 机器人的运动范围 ==============
地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐
标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到
方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，
机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，
因为3+5+3+8=19。请问该机器人能够到达多少个格子？

示例 1：
输入：m = 2, n = 3, k = 1
输出：3

示例 2：
输入：m = 3, n = 1, k = 0
输出：1

提示：
    1 <= n,m <= 100
    0 <= k <= 20

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof
*/
/*
	方法一：自顶向下深度优先搜索【DFS】
	思路：
		我们定义一个 m*n 的二维数组 grid，接着从 grid[0][0] 出发，沿着上下
		左右的路径进行深度优先搜索，在搜索的过程中我们对走过的路径 grid[i][j]
		进行标记，每进行一次标记，我们就让机器人能够到达的格子数 + 1，直到该
		路径已经被标记或无法再走下去的时候返回。
	时间复杂度：O(m*n)
		最坏情况下机器人能够走完所有的格子，此时我们需要标记和统计所有的格子。
	空间复杂度：O(m*n)
		我们需要用一个 m*n 的二维数组来标记机器人能够走到的所有格子。
*/
func movingCount(m int, n int, k int) int {
	if m == 0 || n == 0 {
		return 0
	}
	// 定义和初始化网格
	grid := make([][]bool, m)
	for i := 0; i < m; i++ {
		grid[i] = make([]bool, n)
	}

	// i, j 当前所处的网格下标
	// sum 当前下标的数位和
	// *cnt 可达的格子数
	var DFS func(i, j int, cnt *int)
	DFS = func(i, j int, cnt *int) {
		// 越界或已被标记或数位之和大于 k，说明该位置不可达
		if i < 0 || i >= m || j < 0 || j >= n || grid[i][j] ||
			(getSum(i)+getSum(j)) > k {
			return
		}
		// 标记 grid[i][j] 为可达
		grid[i][j] = true
		// 机器人可达的格子数 +1
		(*cnt)++

		// 向上下左右发散
		DFS(i-1, j, cnt)
		DFS(i+1, j, cnt)
		DFS(i, j-1, cnt)
		DFS(i, j+1, cnt)
	}
	ans := 0
	DFS(0, 0, &ans)
	return ans
}

// 计算数位之和
func getSum(a int) int {
	sum := 0
	for a > 0 {
		sum += a % 10
		a /= 10
	}
	return sum
}

/*
	方法二：自底向上深度优先搜索【DFS】
	思路：
		方法一的结果计算是从顶部传指针然后在每一层中逐一计算的，因为我们
		使用的是递归，所以我们还可以借助于递归返回来计算结果值。
	时间复杂度：O(m*n)
		最坏情况下机器人能够走完所有的格子，此时我们需要标记和统计所有的格子。
	空间复杂度：O(m*n)
		我们需要用一个 m*n 的二维数组来标记机器人能够走到的所有格子。
*/
func movingCount2(m int, n int, k int) int {
	if m == 0 || n == 0 {
		return 0
	}
	// 定义和初始化网格
	grid := make([][]bool, m)
	for i := 0; i < m; i++ {
		grid[i] = make([]bool, n)
	}

	// i, j 当前所处的网格下标
	// sum 当前下标的数位和
	// 返回可达的格子数
	var DFS func(i, j int) int
	DFS = func(i, j int) int {
		// 越界或已被标记或数位之和大于 k，说明该位置不可达
		if i < 0 || i >= m || j < 0 || j >= n || grid[i][j] ||
			(getSum(i)+getSum(j)) > k {
			return 0
		}
		// 标记 grid[i][j] 为可达
		grid[i][j] = true
		// 可达数=当前位置+向其邻居节点发散的可达数
		return 1 + DFS(i-1, j) + DFS(i+1, j) + DFS(i, j-1) + DFS(i, j+1)
	}
	return DFS(0, 0)
}

/*
	方法三：广度优先搜索【BFS】
	思路：
		我们定义一个 m*n 的二维数组 grid，接着从 grid[0][0] 出发，沿着上下
		左右的路径进行广度优先搜索，在搜索的过程中我们对走过的路径 grid[i][j]
		进行标记，每进行一次标记，我们就让机器人能够到达的格子数 + 1，直到该
		路径已经被标记或无法再走下去的时候返回。
	时间复杂度：O(m*n)
		最坏情况下机器人能够走完所有的格子，此时我们需要标记和统计所有的格子。
	空间复杂度：O(m*n)
		我们需要用一个 m*n 的二维数组来标记机器人能够走到的所有格子。
*/
func movingCount3(m int, n int, k int) int {
	if m == 0 || n == 0 {
		return 0
	}
	// 定义和初始化网格
	grid := make([][]bool, m)
	for i := 0; i < m; i++ {
		grid[i] = make([]bool, n)
	}
	queue := make([][2]int, 0)
	// 装入起点坐标并进行标记
	queue = append(queue, [2]int{0, 0})
	grid[0][0] = true
	// 因为 [0,0] 是可达的，所以 ans 初始值为 1
	ans := 1
	for len(queue) > 0 {
		// 取出当前格子的下标
		i, j := queue[0][0], queue[0][1]
		queue = queue[1:]

		// 向上下左右处理 grid[i][j] 的邻居节点
		if i-1 >= 0 && getSum(i-1)+getSum(j) <= k && !grid[i-1][j] {
			queue = append(queue, [2]int{i - 1, j})
			grid[i-1][j] = true
			ans++
		}
		if i+1 < m && getSum(i+1)+getSum(j) <= k && !grid[i+1][j] {
			queue = append(queue, [2]int{i + 1, j})
			grid[i+1][j] = true
			ans++
		}
		if j-1 >= 0 && getSum(i)+getSum(j-1) <= k && !grid[i][j-1] {
			queue = append(queue, [2]int{i, j - 1})
			grid[i][j-1] = true
			ans++
		}
		if j+1 < n && getSum(i)+getSum(j+1) <= k && !grid[i][j+1] {
			queue = append(queue, [2]int{i, j + 1})
			grid[i][j+1] = true
			ans++
		}
	}
	return ans
}

func main() {
	m, n, k := 11, 8, 16
	// res := movingCount(m, n, k)
	res := movingCount2(m, n, k)
	fmt.Println(res)
}
