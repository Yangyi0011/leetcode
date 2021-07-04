package main

import (
	"fmt"
)

/* 
递归和动规关系
	递归：是一种程序的实现方式：函数的自我调用
		Function(x) {
			...
			Funciton(x-1);
			...
		}
	动态规划：是一种解决问题的思想，大规模问题的结果，是由小规模问题的结果运算得来的。动态规划可用递归来实现(Memorization Search)

使用场景
	满足两个条件
		满足以下条件之一
			求最大/最小值（Maximum/Minimum ）
			求是否可行（Yes/No ）
			求可行个数（Count(*) ）
		满足不能排序或者交换（Can not sort / swap ）
		如题：longest-consecutive-sequence  位置可以交换，所以不用动态规划

四点要素
    状态 State
        灵感，创造力，存储小规模问题的结果
    方程 Function
        状态之间的联系，怎么通过小的状态，来算大的状态
    初始化 Intialization
        最极限的小状态是什么, 起点
    答案 Answer
        最大的那个状态是什么，终点

常见四种类型
    Matrix DP (10%)
    Sequence (40%)
    Two Sequences DP (40%)
    Backpack (10%)

注意点
	贪心算法大多题目靠背答案，所以如果能用动态规划就尽量用动规，不用贪心算法
*/

/* 
===================== 1、三角形最小路径和 =====================
给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。
相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。

例如，给定三角形：
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]

自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。

说明：
如果你可以只使用 O(n) 的额外空间（n 为三角形的总行数）来解决这个问题，那么你的算法会很加分。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/triangle
*/
/* 
	方法一：DFS深度优先遍历-自顶向下
	思路：
		使用 DFS 从 triangle[0][0] 开始，以 i = 0，通过 i 和 i + 1 的路径往下遍历，
		达到最深处的时候取路径和的最小值。
	时间复杂度：O(n^2)
		n 表示二维数组的元素个数
	空间复杂度：O(logn)
		递归遍历需要 O(logn) 的额外栈空间
	注：结果超时
*/
func minimumTotal1(triangle [][]int) int {
	n := len(triangle)
	if n == 0 || len(triangle[0]) == 0 {
		return 0
	}
	best := 1 << 31 - 1
	var dfs func(int, int, int)
	dfs = func(x, y, sum int) {
		if x == n {
			if sum < best {
				best = sum
			}
			return
		}
		dfs(x + 1, y, sum + triangle[x][y])
		dfs(x + 1, y + 1, sum + triangle[x][y])
	}
	dfs(0, 0, 0)
	return best
}

/* 
	方法二：DFS分治法-自底向上
	思路：
		从 x, y 出发走到最底层所能找到的最小路径和
	时间复杂度：O(n^2)
		n 表示二维数组的元素个数
	空间复杂度：O(logn)
		递归遍历需要 O(logn) 的额外栈空间
	注：结果超时
*/
func minimumTotal2(triangle [][]int) int {
	n := len(triangle)
	if n == 0 || len(triangle[0]) == 0 {
		return 0
	}
	var dfs func(int, int) int
	dfs = func(x, y int) int {
		if x == n {
			return 0
		}
		return min(dfs((x + 1), y), dfs(x+1, y + 1)) + triangle[x][y]
	}
	return dfs(0, 0)
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
/* 
	方法三：DFS记忆优化
	思路：
		优化 DFS，缓存已经被计算的值，即记忆化搜索，本质上是动态规划。
*/
func minimumTotal3(triangle [][]int) int {
	n := len(triangle)
	if n == 0 || len(triangle[0]) == 0 {
		return 0
	}
	// 初始化表
	hash := make([][]int, 0)
	for i := 0; i < n; i ++ {
		cn := len(triangle[i])
		hash = append(hash, make([]int, cn))
		for j := 0; j < cn; j ++ {
			hash[i][j] = -1
		}
	}
	var dfs func(int, int) int
	dfs = func(x, y int) int {
		if x == n {
			return 0
		}
		// -1 表示还没被计算出来
		if hash[x][y] != -1 {
			return hash[x][y]
		}
		// 打表
		hash[x][y] = min(dfs(x + 1, y), dfs(x + 1, y + 1)) + triangle[x][y]
		return hash[x][y]
	}
	return dfs(0, 0)
}

/* 
	方法四：DP
	思路：
		动态规划，自底向上
	
*/
func minimumTotal4(triangle [][]int) int {
	if len(triangle) == 0 || len(triangle[0]) == 0 {
		return 0
	}
	// 1、状态定义：f[i][j] 表示从i,j出发，到达最后一层的最短路径
	var l = len(triangle)
	var f = make([][]int, l)
	// 2、初始化
	for i := 0; i < l; i++ {
		for j := 0; j < len(triangle[i]); j++ {
			if f[i] == nil {
				f[i] = make([]int, len(triangle[i]))
			}
			f[i][j] = triangle[i][j]
		}
	}
	// 3、递推求解
	for i := len(triangle) - 2; i >= 0; i-- {
		for j := 0; j < len(triangle[i]); j++ {
			f[i][j] = min(f[i+1][j], f[i+1][j+1]) + triangle[i][j]
		}
	}
	// 4、答案
	return f[0][0]
}

/* 
	方法五：DP
	思路：
		动态规划，自顶向下
*/
func minimumTotal5(triangle [][]int) int {
    if len(triangle) == 0 || len(triangle[0]) == 0 {
        return 0
    }
    // 1、状态定义：f[i][j] 表示从0,0出发，到达i,j的最短路径
    var l = len(triangle)
    var f = make([][]int, l)
    // 2、初始化
    for i := 0; i < l; i++ {
        for j := 0; j < len(triangle[i]); j++ {
            if f[i] == nil {
                f[i] = make([]int, len(triangle[i]))
            }
            f[i][j] = triangle[i][j]
        }
    }
    // 递推求解
    for i := 1; i < l; i++ {
        for j := 0; j < len(triangle[i]); j++ {
            // 这里分为两种情况：
            // 1、上一层没有左边值
            // 2、上一层没有右边值
            if j-1 < 0 {
                f[i][j] = f[i-1][j] + triangle[i][j]
            } else if j >= len(f[i-1]) {
                f[i][j] = f[i-1][j-1] + triangle[i][j]
            } else {
                f[i][j] = min(f[i-1][j], f[i-1][j-1]) + triangle[i][j]
            }
        }
    }
    result := f[l-1][0]
    for i := 1; i < len(f[l-1]); i++ {
        result = min(result, f[l-1][i])
    }
    return result
}

/* 
	算法：常规DP
	思路：
		我们用 f[i][j] 表示从三角形顶部走到位置 (i,j) 的最小路径和。这里的位置 (i,j) 
		指的是三角形中第 i 行第 j 列（均从 0 开始编号）的位置。
		由于每一步只能移动到下一行「相邻的节点」上，因此要想走到位置 (i,j)，
		上一步就只能在位置 (i−1,j−1) 或者位置 (i−1,j)。
		我们在这两个位置中选择一个路径和较小的来进行转移，状态转移方程为：
			f[i][j]=min⁡(f[i−1][j−1],f[i−1][j])+c[i][j]
		其中 c[i][j] 表示位置 (i,j) 对应的元素值。
		注意第 i 行有 i+1 个元素，它们对应的 j 的范围为 [0,i]。当 j=0 或 j=i 时，
		上述状态转移方程中有一些项是没有意义的。例如当 j=0 时，f[i−1][j−1] 没有意义，
		因此状态转移方程为：
			f[i][0]=f[i−1][0]+c[i][0]
		即当我们在第 i 行的最左侧时，我们只能从第 i−1 行的最左侧移动过来。
		当 j=i 时，f[i−1][j] 没有意义，因此状态转移方程为：
			f[i][i]=f[i−1][i−1]+c[i][i]
		即当我们在第 i 行的最右侧时，我们只能从第 i−1 行的最右侧移动过来。
		最终的答案即为 f[n−1][0] 到 f[n−1][n−1] 中的最小值，其中 n 是三角形的行数。
	细节：
		状态转移方程的边界条件是什么？由于我们已经去除了所有「没有意义」的状态，
		因此边界条件可以定为：
			f[0][0]=c[0][0]
		即在三角形的顶部时，最小路径和就等于对应位置的元素值。这样一来，
		我们从 1 开始递增地枚举 i，并在 [0,i] 的范围内递增地枚举 j，
		就可以完成所有状态的计算。
	时间复杂度：O(n^2)
		其中 n 是三角形的行数。
	空间复杂度：O(n^2)
		其中 n 是三角形的行数，我们需要一个 n∗n 的二维数组存放所有的状态。
*/
func minimumTotal6(triangle [][]int) int {
	if len(triangle) == 0 || len(triangle[0]) == 0 {
		return 0
	}
	n := len(triangle)
	dp := make([][]int, n)
	for i := 0; i < n; i ++ {
		dp[i] = make([]int, len(triangle[i]))
	}
	// 打表，设置初始值
	dp[0][0] = triangle[0][0]
	for i := 1; i < n; i ++ {
		// 当前层的最左边的必定是从上一层的最左边来的
		dp[i][0] = dp[i - 1][0] + triangle[i][0]
		// 每一层的元素个数等于其层数+1
		for j := 1; j < i; j ++ {
			dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle[i][j]
		}
		// 当前层的最右边必定是从上一层的最后一个来的
		dp[i][i] = dp[i - 1][i - 1] + triangle[i][i]
	}
	// 由于是自顶向下的打表，所以完成后，答案在表的最后一层，只需遍历找出最小值就行了
	ans := dp[n - 1][0]
	for i := 1; i < len(dp[n - 1]); i ++ {
		ans = min(ans, dp[n - 1][i])
	} 
	return ans
}

/* 	
	算法：常规DP + 空间优化
	思路：
		在题目描述中的「说明」部分，提到了可以将空间复杂度优化至 O(n)。
		我们回顾方法一中的状态转移方程：
					   f[i−1][0]+c[i][0],						j=0
			f[i][j]= / f[i−1][i−1]+c[i][i],						j=i
					 \ min⁡(f[i−1][j−1],f[i−1][j])+c[i][j],		 otherwise

		可以发现，f[i][j] 只与 f[i−1][..] 有关，而与 f[i−2] 及之前的状态无关，
		因此我们不必存储这些无关的状态。具体地，我们使用两个长度为 n 的一维数组进行转移，
		将 i 根据奇偶性映射到其中一个一维数组，那么 i−1 就映射到了另一个一维数组。
		这样我们使用这两个一维数组，交替地进行状态转移。
	时间复杂度：O(n^2)
		其中 n 是三角形的行数。
	空间复杂度：O(2n)
		其中 n 是三角形的行数，我们需要两个长度为 n 的数组存放上一层的状态和当前层的状态。
*/
func minimumTotal7(triangle [][]int) int {
	if len(triangle) == 0 || len(triangle[0]) == 0 {
		return 0
	}
	n := len(triangle)
	dp := make([][]int, 2)
	for i := 0; i < 2; i ++ {
		dp[i] = make([]int, n)
	}
	dp[0][0] = triangle[0][0]
	for i := 1; i < n; i ++ {
		cur := i & 1	// 等价于 i % 2
		pre := 1 - cur
		dp[cur][0] = dp[pre][0] + triangle[i][0]
		for j := 1; j < i; j ++ {
			dp[cur][j] = min(dp[pre][j - 1], dp[pre][j]) + triangle[i][j]
		}
		dp[cur][i] = dp[pre][i - 1] + triangle[i][i]
	}
	// 用以确定最后的结果保存在哪一行，层数为奇结果保存在第 1 行，层数为偶结果保存在第 0 行
	row := (n - 1) & 1
	ans := dp[row][0]
	for i := 1; i < n; i ++ {
		ans = min(ans, dp[row][i])
	}
	return ans
}

/* 	
	算法：常规DP + 空间优化 + 继续优化
	思路：
		上述方法的空间复杂度为 O(n)，使用了 2n 的空间存储状态。我们还可以继续进行优化吗？
		答案是可以的。我们从 i 到 0 递减地枚举 j，这样我们只需要一个长度为 n 的一维数组 f，就可以完成状态转移。
		为什么只有在递减地枚举 j 时，才能省去一个一维数组？
		当我们在计算位置 (i,j) 时，f[j+1] 到 f[i] 已经是第 i 行的值，而 f[0] 到 f[j] 仍然是第 i−1 行的值。
		此时我们直接通过
			f[j]=min⁡(f[j−1],f[j])+c[i][j]
		进行转移，恰好就是在 (i−1,j−1) 和 (i−1,j) 中进行选择。但如果我们递增地枚举 j，那么在计算位置 (i,j) 时，
		f[0] 到 f[j−1] 已经是第 i 行的值。如果我们仍然使用上述状态转移方程，那么是在 (i,j−1) 和 (i−1,j) 中进行选择，就产生了错误。
		这样虽然空间复杂度仍然为 O(n)，但我们只使用了 n 的空间存储状态，减少了一半的空间消耗。
	时间复杂度：O(n^2)
		其中 n 是三角形的行数。
	空间复杂度：O(n)
		其中 n 是三角形的行数，我们需要一个长度为 n 的数组存放上一层的状态和当前层的状态。
*/
func minimumTotal8(triangle [][]int) int {
	if len(triangle) == 0 || len(triangle[0]) == 0 {
		return 0
	}
	n := len(triangle)
	dp := make([]int, n)
	dp[0] = triangle[0][0]
	for i := 1; i < n; i ++ {
		dp[i] = dp[i - 1] + triangle[i][i]
		for j :=  i - 1; j > 0; j -- {
			dp[j] = min(dp[j - 1], dp[j]) + triangle[i][j]
		}
		dp[0] += triangle[i][0]
	}
	ans := dp[0]
	for i := 1; i < n; i ++ {
		ans = min(ans, dp[i])
	}
	return ans
}

/* 
================== 2、最长连续序列 ==================
给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

进阶：你可以设计并实现时间复杂度为 O(n) 的解决方案吗？

示例 1：
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。

示例 2：
输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9

提示：
    0 <= nums.length <= 104
    -109 <= nums[i] <= 109
来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/longest-consecutive-sequence
*/

/* 
	方法一：哈希表
	思路和算法
		我们考虑枚举数组中的每个数 x，考虑以其为起点，不断尝试匹配 x+1,x+2,⋯ 是否存在，
		假设最长匹配到了 x+y，那么以 x 为起点的最长连续序列即为 x,x+1,x+2,⋯ ,x+y，
		其长度为 y+1，我们不断枚举并更新答案即可。
		对于匹配的过程，暴力的方法是 O(n) 遍历数组去看是否存在这个数，
		但其实更高效的方法是用一个哈希表存储数组中的数，这样查看一个数是否存在即能优化
		至 O(1) 的时间复杂度。
		仅仅是这样我们的算法时间复杂度最坏情况下还是会达到 O(n^2)（即外层需要枚举 O(n) 个数，
		内层需要暴力匹配 O(n) 次），无法满足题目的要求。但仔细分析这个过程，
		我们会发现其中执行了很多不必要的枚举，如果已知有一个 x,x+1,x+2,⋯ ,x+y 的连续序列，
		而我们却重新从 x+1，x+2 或者是 x+y 处开始尝试匹配，那么得到的结果肯定不会优于
		枚举 x 为起点的答案，因此我们在外层循环的时候碰到这种情况跳过即可。
		那么怎么判断是否跳过呢？由于我们要枚举的数 x 一定是在数组中不存在前驱数 x−1 的，
		不然按照上面的分析我们会从 x−1 开始尝试匹配，因此我们每次在哈希表中检查是否
		存在 x−1 即能判断是否需要跳过了。
	时间复杂度：O(n)
		其中 n 为数组的长度。具体分析已在上面正文中给出。
	空间复杂度：O(n)
		哈希表存储数组中所有的数需要 O(n) 的空间。
*/
func longestConsecutive(nums []int) int {
    numSet := map[int]bool{}
    for _, num := range nums {
        numSet[num] = true
    }
    longestStreak := 0
    for num, _ := range numSet {
        if !numSet[num-1] {
            currentNum := num
            currentStreak := 1
            for numSet[currentNum+1] {
                currentNum++
                currentStreak++
            }
            if longestStreak < currentStreak {
                longestStreak = currentStreak
            }
        }
    }
    return longestStreak
}

/* 
================== 3、最小路径和 ==================
给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，
使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

示例 1：
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。

示例 2：
输入：grid = [[1,2,3],[4,5,6]]
输出：12

提示：
    m == grid.length
    n == grid[i].length
    1 <= m, n <= 200
    0 <= grid[i][j] <= 100

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/minimum-path-sum
*/
/* 
	方法一：原始DP
	思路：
		我们需要一个二维数组dp[m][n]来存储每一个计算的值（状态）。
		初始状态：
			我们是从左上角开始走的，故初始状态为：
				dp[0][0] = grid[0][0]
		中间状态：
			对于想要到达的位置dp[i][j]，依题意就只能从 dp[i][j-1]（左边）或 dp[i-1][j]（上边）过来，
			故此时的状态转移方程为：
				dp[i][j] = min(dp[i][j-1], dp[i-1][j]) + grid[i][j]
		边界条件：
			1、我们知道第一行的元素除了第一个是初始状态外，想要到达其余元素就只能从左边来，
				故第一行元素的状态转移方程为：
					dp[0][j] = dp[0][j-1] + grid[0][j], j∈[1,n)
			2、对于最左边的元素，除了dp[0][0]外，想要到达它的位置，就只能从上一行的相同位置过来，
				故最左边元素的状态转移方程为：
					dp[i][0] = dp[i-1][0] + grid[i][0], i∈[1,m)
		终止状态：
			我们需要走到右下角，故终止状态为：
				dp[m-1][n-1]
		综上所述，我们得到状态转移方程如下：
					  / grid[i][j], i=0, j=0
			dp[i][j] /  dp[i][j-1] + grid[i][j], i=0,j∈[1,n)
						dp[i-1][j] + grid[i][j], i∈[1,m), j=0
					 \	min(dp[i][j-1],dp[i-1][j]) + grid[i][j], i∈[1,m), j∈[1,n)
	时间复杂度：O(m*n)
	空间复杂度：O(m*n)
*/
func minPathSum(grid [][]int) int {
	n := len(grid)
	if n == 0 || len(grid[0]) == 0 {
		return 0
	}
	// 用 dp[i][j] 来表示 从 grid[0][0] 到 grid[i][j] 的最小距离和
	dp := make([][]int, n)
	for i := 0; i < n; i ++ {
		dp[i] = make([]int, len(grid[i]))
	}
	// 初始化
	dp[0][0] = grid[0][0]
	// 第0行的状态只能从左边来
	for i := 1; i < len(grid[0]); i ++ {
		dp[0][i] = dp[0][i - 1] + grid[0][i]
	}
	for i := 1; i < n; i ++ {
		// 在行的最左边，那就只能从上一行的最左边过来
		dp[i][0] = dp[i - 1][0] + grid[i][0]
		for j := 1; j < len(grid[i]); j ++ {
			// 可能是从上边来的，也可能是从左边来的
			dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
		}
	}
	// 目的地在最后一行的最后一个元素
	return dp[n - 1][len(dp[n - 1]) - 1]
}

/* 
	方法二：DP-空间优化
	思路：
		通过上面的dp我们已经知道，dp[i][j]只跟左边的元素dp[i][j-1]和上边的元素dp[i-1][j]
		有关，所以我们不需要记录那么多的状态，而我们的最终目的是要走到dp[m-1][n-1]的，
		即走到最后一行，故我们只需两个长度为 n 的数组来记录当前行和上一行的状态，再通过奇偶来判断当前行
		的状态该放到哪一个数组中就行了。
	时间复杂度：O(m*n)
	空间复杂度：O(2n)
*/
func minPathSum2(grid [][]int) int {
	n := len(grid)
	if n == 0 || len(grid[0]) == 0 {
		return 0
	}
	dp := make([][]int, 2)
	for i := 0; i < 2; i ++ {
		dp[i] = make([]int, len(grid[0]))
	}
	// 初始化
	dp[0][0] = grid[0][0]
	// 第0行的状态只能从左边来
	for i := 1; i < len(grid[0]); i ++ {
		dp[0][i] = dp[0][i - 1] + grid[0][i]
	}
	for i := 1; i < n; i ++ {
		cur := i & 1 // 同 i % 2
		pre := 1 - cur
		// 在行的最左边，那就只能从上一行的最左边过来
		dp[cur][0] = dp[pre][0] + grid[i][0]
		for j := 1; j < len(grid[i]); j ++ {
			// 可能是从上边来的，也可能是从左边来的
			dp[cur][j] = min(dp[pre][j], dp[cur][j - 1]) + grid[i][j]
		}
	}
	// 寻找最后一行的状态在哪一个数组中
	row := (n-1) & 1	// 同 (n-1)%2
	// 目的地在最后一行的最后一个元素
	return dp[row][len(dp[row]) - 1]
}

/* 
	方法二：DP-空间优化-续
	思路：
		通过上一次的空间优化，我们已经把空间复杂度优化到2n了，能不能优化到n呢？
		我们知道每一个位置的状态只能基于左边或是上边的状态来改变，如此当我们只用
		一个数组来存储上一行的状态时，我们如果在当前行按从左往右的顺序做状态改变，
		那么是不会影响到当前行后面位置的状态的，因为每一个位置的状态改变都只能基
		于上边的位置或是左边的位置的状态，如果dp[i]是基于左边的状态来改变自己状态的，
		那么这个左边的状态必定是当前行的且已经改变过了的状态，符合我们从左到右的
		状态改变顺序。如果dp[i]是基于上一行的位置来做状态改变的，那么我们从左到
		右改变状态时，dp[i]的位置存储的就是上一行的状态，并没有被提前修改，我们只
		需要基于上一行dp[i]的状态来改变当前行dp[i]的状态即可。
		由此我们就可以只依靠一个长度为n的数组来完成状态转移了。
	时间复杂度：O(m*n)
	空间复杂度：O(n)
*/
func minPathSum3(grid [][]int) int {
	n := len(grid)
	if n == 0 || len(grid[0]) == 0 {
		return 0
	}
	dp := make([]int, len(grid[0]))
	// 初始化
	dp[0] = grid[0][0]
	// 第0行的状态只能从左边来
	for i := 1; i < len(grid[0]); i ++ {
		dp[i] = dp[i - 1] + grid[0][i]
	}
	for i := 1; i < n; i ++ {
		// 在行的最左边，那就只能从上一行的最左边过来
		dp[0] = dp[0] + grid[i][0]
		for j := 1; j < len(grid[i]); j ++ {
			// 可能是从上边来的，也可能是从左边来的
			dp[j] = min(dp[j], dp[j - 1]) + grid[i][j]
		}
	}
	// 目的地在最后一行的最后一个元素
	return dp[len(dp) - 1]
}

/* 
================== 4、不同路径 ==================
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
问总共有多少条不同的路径？

示例 1:

输入: m = 3, n = 2
输出: 3
解释:
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向右 -> 向下
2. 向右 -> 向下 -> 向右
3. 向下 -> 向右 -> 向右

示例 2:

输入: m = 7, n = 3
输出: 28

提示：
    1 <= m, n <= 100
    题目数据保证答案小于等于 2 * 10 ^ 9

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/unique-paths
*/
/* 
	方法一：原始DP
	思路：
		我们需要一个二维数组dp[m][n]来存储每一个网格的路径数（状态）。
		dp[i][j] 表示从 grid[0][0] 到达 grid[i][j] 的路径总数
		初始状态：
			我们是从左上角开始走的，故初始状态为：
				dp[0][0] = 1  表示到达 grid[0][0] 位置只有一条路径
		中间状态：
			对于想要到达的位置dp[i][j]，依题意就只能从 dp[i][j-1]（左边）或 dp[i-1][j]（上边）过来，
			故此时的状态转移方程为：
				dp[i][j] = dp[i][j-1] + dp[i-1][j]
		边界条件：
			1、我们知道第一行的元素，除了dp[0][0]外，想要到达其余元素就只能从左边来，
				故第一行元素的状态转移方程为：
					dp[0][j] = dp[0][j-1], j∈[1,n)
			2、对于最左边的元素，除了dp[0][0]外，想要到达它的位置，就只能从上一行的相同位置过来，
				故最左边元素的状态转移方程为：
					dp[i][0] = dp[i-1][0], i∈[1,m)
		终止状态：
			我们需要走到右下角，故终止状态为：
				dp[m-1][n-1]
		综上所述，我们得到状态转移方程如下：
					  / 1, i=0, j=0
			dp[i][j] /  dp[i][j-1], i=0,j∈[1,n)
						dp[i-1][j], i∈[1,n),j=0
					 \	dp[i][j-1] + dp[i-1][j], i∈[1,m), j∈[1,n)
	时间复杂度：O(m*n)
	空间复杂度：O(m*n)
*/
func uniquePaths(m int, n int) int {
	if m == 0 || n == 0 {
		return 0
	}
	// dp[i][j] 表示i,j到0,0路径数
	dp := make([][]int, m)
	for i := 0; i < m; i ++ {
		dp[i] = make([]int, n)
	}
	// 初始化
	dp[0][0] = 1
	// 第一行只能从左边过来
	for j := 1; j < n; j ++ {
		dp[0][j] = dp[0][j-1]
	}
	for i := 1; i < m; i++ {
		// 第一列只能从上边过来
		dp[i][0] = dp[i-1][0]
		for j := 1; j < n; j++ {
			// 其余的可能从左边，也可能从右边
			dp[i][j] = dp[i][j-1] + dp[i-1][j]
		}
	}
	return dp[m-1][n-1]
}

/* 
	方法二：DP-空间优化
	思路：
		从方法一中可知，对于每一个位置的路径总数dp[i][j]，只受左边和上边位置的状态影响，
		故我们可以只用两个长度为 n 的数组来记录上一行的状态和当前行的状态，再通过奇偶
		性来确定每一行的状态该放在哪一个数组中，如此就可以把空间复杂度优化到O(2n)了。
	时间复杂度：O(m*n)
	空间复杂度：O(2n)
*/
func uniquePaths2(m int, n int) int {
	if m == 0 || n == 0 {
		return 0
	}
	dp := make([][]int, 2)
	for i := 0; i < 2; i ++ {
		dp[i] = make([]int, n)
	}
	// 初始化
	dp[0][0] = 1
	for j := 1; j < n; j ++ {
		dp[0][j] = dp[0][j-1]
	}
	for i := 1; i < m; i ++ {
		cur := i & 1 // 同 i%2
		pre := 1 - cur
		dp[cur][0] = dp[pre][0]
		for j := 1; j < n; j ++ {
			dp[cur][j] = dp[cur][j-1] + dp[pre][j]
		}
	}
	// 寻找最后一行的状态在哪一个数组中
	row := (m-1) & 1	// 同 (m-1)%2
	// 目的地在最后一行的最后一个元素
	return dp[row][n - 1]
}

/* 
	方法二：DP-空间优化-续
	思路：
		通过上一次的空间优化，我们已经把空间复杂度优化到2n了，能不能优化到n呢？
		我们知道每一个位置的状态只能基于左边或是上边的状态来改变，如此当我们只用
		一个数组来存储上一行的状态时，我们如果在当前行按从左往右的顺序做状态改变，
		那么是不会影响到当前行后面位置的状态的，因为每一个位置的状态改变都只能基
		于上边的位置或是左边的位置的状态，如果dp[i]是基于左边的状态来改变自己状态的，
		那么这个左边的状态必定是当前行的且已经改变过了的状态，符合我们从左到右的
		状态改变顺序。如果dp[i]是基于上一行的位置来做状态改变的，那么我们从左到
		右改变状态时，dp[i]的位置存储的就是上一行的状态，并没有被提前修改，我们只
		需要基于上一行dp[i]的状态来改变当前行dp[i]的状态即可。
		由此我们就可以只依靠一个长度为n的数组来完成状态转移了。
	时间复杂度：O(m*n)
	空间复杂度：O(n)
*/
func uniquePaths3(m int, n int) int {
	if m == 0 || n == 0 {
		return 0
	}
	dp := make([]int, n)
	// 初始化
	dp[0] = 1
	for j := 1; j < n; j ++ {
		dp[j] = dp[j-1]
	}
	for i := 1; i < m; i ++ {
		// dp[0] = dp[0] 最左边元素的处理直接省略，因为都是 1
		for j := 1; j < n; j ++ {
			dp[j] = dp[j-1] + dp[j]
		}
	}
	// 目的地在最后一行的最后一个元素
	return dp[n - 1]
}

/* 
================== 5、不同路径 II ==================
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
网格中的障碍物和空位置分别用 1 和 0 来表示。

示例 1：
输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
输出：2

解释：
3x3 网格的正中间有一个障碍物。
从左上角到右下角一共有 2 条不同的路径：
1. 向右 -> 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右 -> 向右

示例 2：
输入：obstacleGrid = [[0,1],[0,0]]
输出：1

提示：
    m == obstacleGrid.length
    n == obstacleGrid[i].length
    1 <= m, n <= 100
    obstacleGrid[i][j] 为 0 或 1

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/unique-paths-ii
*/
/* 
	方法一：初始DP
	思路：
		路径总数状态转移的计算方式与《不同路径I》一样，只是额外增加了障碍物的边界条件，
		我们只需要考虑在遇到障碍物时该怎么处理就行了。
		对于目标位置 grid[m-1][n-1]：
			如果该位置被障碍物占了，则无论如何都不可达，直接返回0就行了。
		对于第一行:
			如果不存在障碍物，则：
				dp[0][j] = dp[0][j-1], j∈[1,n)
			如果有障碍物存在，则障碍物后面的位置都将变得不可达，即：
				当 grid[0][j-1]==1 时，dp[0][j] = 0, j∈[1,n)
		对于第一列：
			如果不存在障碍物，则：
				dp[i][0] = dp[i-1][0], i∈[1,m)
			如果有障碍物存在，则障碍物下面的位置都将变得不可达，即：
				当 grid[i][0]==1 时，dp[i][0] = 0, i∈[1,m)
		对于中间位置grid[i][j]:
			如果上边和左边都不存在障碍物，则路径数等于左边的路径数+上边的路径数，即：
				当 grid[i][j-1] != 1 && grid[i-1][j] != 1 时，
					dp[i][j] = dp[i][j-1] + dp[i-1][j]
			否则：
				1、左边没有障碍物：
					dp[i][j] = dp[i][j-1]
				2、上边没有障碍物：
					dp[i][j] = dp[i-1][j]
				3、左边和上边都有障碍物
					dp[i][j] = 0
		最终结果保存在 dp[m-1][n-1] 中
	时间复杂度：O(m*n)
	空间复杂度：O(m*n)
*/
func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	m := len(obstacleGrid)
	if m == 0 || len(obstacleGrid[0]) == 0 {
		return 0
	}
	n := len(obstacleGrid[0])
	// 如果目标位置有障碍物，则不可达
	if obstacleGrid[m-1][n-1] == 1 {
		return 0
	}
	dp := make([][]int, m)
	for i := 0; i < m; i ++ {
		dp[i] = make([]int, n)
	}
	dp[0][0] = 1
	// 第一行若是有障碍物的话，则障碍物后面的位置都不可达
	for j := 1; j < n; j ++ {
		// 有障碍物
		if obstacleGrid[0][j-1] == 1 {
			dp[0][j] = 0
		} else {
			dp[0][j] = dp[0][j-1]
		}
	}
	for i := 1; i < m; i ++ {
		// 第一列若是有障碍物的话，则障碍物下面的位置都不可达
		if obstacleGrid[i-1][0] == 1 {
			dp[i][0] = 0
		} else {
			dp[i][0] = dp[i-1][0]
		}
		for j := 1; j < n; j ++ {
			// 上边和左边都没有障碍物
			if obstacleGrid[i][j-1] != 1 && obstacleGrid[i-1][j] != 1 {
				dp[i][j] = dp[i][j-1] + dp[i-1][j]
			} else {
				if obstacleGrid[i][j-1] == 1 && obstacleGrid[i-1][j] == 1 {
					// 上边和左边都有障碍物
					dp[i][j] = 0
				} else if obstacleGrid[i][j-1] != 1 {
					// 左边没有障碍物
					dp[i][j] = dp[i][j-1]
				} else {
					// 上边没有障碍物
					dp[i][j] = dp[i-1][j]
				}
			}
		}
	}
	return dp[m-1][n-1]
}

/* 
	方法二：DP-空间优化
	思路：
		此题我们可以将空间复杂度优化到 O(n) 吗？
		答案与题目《不同路径I》一样是可以的，我们依旧只需要一个数组就能完成状态转移，
		只需要注意由障碍物产生的边界就行了。
	时间复杂度：O(m*n)
	空间复杂度：O(n)
*/
func uniquePathsWithObstacles2(obstacleGrid [][]int) int {
	m := len(obstacleGrid)
	if m == 0 || len(obstacleGrid[0]) == 0 {
		return 0
	}
	n := len(obstacleGrid[0])
	// 如果目标位置有障碍物，则不可达
	if obstacleGrid[m-1][n-1] == 1 {
		return 0
	}
	dp := make([]int, n)
	dp[0] = 1
	// 第一行若是有障碍物的话，则障碍物后面的位置都不可达
	for j := 1; j < n; j ++ {
		// 有障碍物
		if obstacleGrid[0][j-1] == 1 {
			dp[j] = 0
		} else {
			dp[j] = dp[j-1]
		}
	}
	for i := 1; i < m; i ++ {
		// 第一列若是有障碍物的话，则障碍物下面的位置都不可达
		if obstacleGrid[i-1][0] == 1 {
			dp[0] = 0
		} // 省略了 else : dp[0] = dp[0]
		
		for j := 1; j < n; j ++ {
			// 上边和左边都没有障碍物
			if obstacleGrid[i][j-1] != 1 && obstacleGrid[i-1][j] != 1 {
				dp[j] = dp[j-1] + dp[j]
			} else {
				if obstacleGrid[i][j-1] == 1 && obstacleGrid[i-1][j] == 1 {
					// 上边和左边都有障碍物
					dp[j] = 0
				} else if obstacleGrid[i][j-1] != 1 {
					// 左边没有障碍物
					dp[j] = dp[j-1]
				} 
				// 上边没有障碍物，不用处理
				// 省略了 else : dp[j] = dp[j]
			}
		}
	}
	return dp[n-1]
}

/* 
================== 6、爬楼梯 ==================
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
注意：给定 n 是一个正整数。

示例 1：
输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。
1.  1 阶 + 1 阶
2.  2 阶

示例 2：
输入： 3
输出： 3
解释： 有三种方法可以爬到楼顶。
1.  1 阶 + 1 阶 + 1 阶
2.  1 阶 + 2 阶
3.  2 阶 + 1 阶

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/climbing-stairs
*/
/* 
	方法一：初始DP
	思路：
		我们用一个长度为 n 的数组来记录状态转移过程。
		初始状态：
			刚开始我们是在楼梯前的，还没上楼梯，此时只有一种状态，即
				dp[0] = 1
		中间状态：
			对于每一个位置 i，它只能从走一步 i-1 或是走两步 i-2 来到达，故有：
				dp[i] = dp[i-1] + dp[i-2]
		边界条件：
			当只有 i = 1 时，我们只有走一步这种一种走法，故有：
				dp[1] = 1
		终止状态：
			我们需要走到楼梯顶部，即 dp[n]
		至此，整个问题其实就是一个 斐波那契数列 问题。
	时间复杂度：O(n)
	空间复杂度：O(n)
*/
func climbStairs(n int) int {
	if n == 0 {
		return 1
	}
	if n == 1 {
		return 1
	}
	// 因为我们把在楼梯前，即还没走楼梯的状态考虑进去了，所以是 n + 1
	dp := make([]int, n + 1)
	dp[0], dp[1] = 1, 1
	for i := 2; i <= n; i ++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

/* 
	方法二：DP-空间优化
	思路：
		由方法一可知，对于每一个位置 i，它只会受到 i-1 和 i-2 两个位置的状态所影响，
		所以我们无需记录所有的状态，只需要用一个长度为 2 的数组来记录 i-1 和 i-2 两个
		位置的状态，再借助于奇偶性来确定每一个状态该放在哪里就行了。
	时间复杂度：O(n)
	空间复杂度：O(1)
*/
func climbStairs2(n int) int {
	if n == 0 {
		return 1
	}
	if n == 1 {
		return 1
	}
	dp := make([]int, 2)
	dp[0], dp[1] = 1, 1
	for i := 2; i <= n; i ++ {
		cur := i & 1 // 同 i % 2
		pre := 1 - cur
		dp[cur] = dp[pre] + dp[cur]
	}
	// 确定最后一个状态存放在哪里，因为总的状态转移次数为 n + 1，
	// dp 数组的下标从 0 开始，所以最后的状态在 dp[n&1] 中
	index := n & 1
	return dp[index]
}

/* 
================== 7、跳跃游戏 ==================
给定一个非负整数数组，你最初位于数组的第一个位置。
数组中的每个元素代表你在该位置可以跳跃的最大长度。
判断你是否能够到达最后一个位置。

示例 1:
输入: [2,3,1,1,4]
输出: true
解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。

示例 2:
输入: [3,2,1,0,4]
输出: false
解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/jump-game
*/

/* 
	方法一：寻找能跳到的最远位置（贪心算法）
	思路：
		设想一下，对于数组中的任意一个位置 y，我们如何判断它是否可以到达？
		根据题目的描述，只要存在一个位置 x，它本身可以到达，并且它跳跃的
		最大长度为 x+nums[x]，这个值大于等于 y，即 x+nums[x]≥y，那么位置 y 
		也可以到达。
		换句话说，对于每一个可以到达的位置 x，它使得 x+1,x+2,⋯ ,x+nums[x] 
		这些连续的位置都可以到达。这样以来，我们依次遍历数组中的每一个位置，
		并实时维护 最远可以到达的位置。对于当前遍历到的位置 x，如果它在 
		最远可以到达的位置 的范围内，那么我们就可以从起点通过若干次跳跃到达
		该位置，因此我们可以用 x+nums[x] 更新 最远可以到达的位置。
		在遍历的过程中，如果 最远可以到达的位置 大于等于数组中的最后一个位置，
		那就说明最后一个位置可达，我们就可以直接返回 True 作为答案。
		反之，如果在遍历结束后，最后一个位置仍然不可达，我们就返回 False 作为答案。
	时间复杂度：O(n)
	空间复杂度：O(1)
*/
func canJump(nums []int) bool {
	n := len(nums)
	if n == 0 { 
		return true
	}
	reach := nums[0]
	/* for i := 1; i < n - 1; i ++ {
		if i > reach {
			// 如果当前位置比前面所有位置跳跃能够到达的最远位置还要大，
			// 说明当前位置不可达
			return false
		}
		if i + nums[i] > reach {
			reach = i + nums[i]
		}
	} */
	// 对上面代码的优化
	for i := 1; i <= reach && reach < n-1; i ++ {
		if i + nums[i] > reach {
			reach = i + nums[i]
		}
	}
	return reach >= n - 1
}

/* 
	方法二：初始DP
	思路：
		我们用一个长度为 n 的数组来存储每一个位置的状态，dp[i] 表示从nums[0]到
		nums[i]是否可达，主要看最后一跳。
		初始状态：
			我们是从nums[0]开始的，即：
				dp[0] = true
		中间状态：
			对于位置 i 是否可达，取决于之前的所有位置的最后一跳是否能到达 i，即：
				dp[i] = dp[j] && j + nums[j] >= i, i ∈ [1,n) j ∈ [0,i)
		终止状态：
			dp[n-1]
		时间复杂度：O(n^2)
			n 表示数组元素个数
		空间复杂度：O(n)
			因为dp[i] 是基于 dp[0] ~ dp[i-1]的，所以空间复杂度无法优化。
*/
func canJump2(nums []int) bool {
	n := len(nums)
	if n == 0 {
		return true
	}
	dp := make([]bool, n)
	dp[0] = true
	for i := 1; i < n; i ++ {
		for j := 0; j < i; j ++ {
			if dp[j] && j + nums[j] >= i {
				dp[i] = true
				// 只要当前位置有一种方式可达就不再做后续判断
				break
			}
		}
	}
	return dp[n-1]
}

/* 
================== 8、跳跃游戏 ==================
给定一个非负整数数组，你最初位于数组的第一个位置。
数组中的每个元素代表你在该位置可以跳跃的最大长度。
你的目标是使用最少的跳跃次数到达数组的最后一个位置。

示例:
输入: [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。

说明:
假设你总是可以到达数组的最后一个位置。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/jump-game-ii
*/
/* 
	方法一：初始DP
	思路：
		用一个长度为 n 的数组来存储每一个位置的状态。
		以 dp[i] 表示从 nums[0] 到达 nums[i] 所需的最少跳跃数
		初始状态：
			在第一个位置时我们不需要跳跃，故有：
				dp[0] = 0
		中间状态：
			假设到达每一个位置 num[i] 时每次只跳一步，则最大需要跳 i 步，
			我们需要在从头寻找到达该位置需要的最少跳跃数，故有：
				dp[i] = i
				dp[i] = min(d[j]+1, dp[i])
		边界条件：
			需要判断每一个位置 nums[i] 是否可以到达，能到达才能更新它的跳跃数，即：
				if j + nums[j] >= i {
					dp[i] = min(d[j]+1, dp[i])
				}
				i∈[0,n), j∈[0,i)
		终止状态：
			我们需要走到最后的位置，即 dp[n-1]
		时间复杂度：O(n^2)
				n 表示数组元素个数
		空间复杂度：O(n)
				因为dp[i] 是基于 dp[0] ~ dp[i-1]的，所以空间复杂度无法优化。
*/
func jump(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	// 以 dp[i] 表示从 nums[0] 到达 nums[i] 所需的最少跳跃次数
	dp := make([]int, n)
	dp[0] = 0
	for i := 1; i < n; i ++ {
		dp[i] = i 	// 到达 nums[i] 最大需要跳 i 次
		for j := 0; j < i; j ++ {
			if j + nums[j] >= i {
				// 寻找需要的最少步数
				dp[i] = min(dp[j] + 1, dp[i])
			}
		}
	}
	return dp[n-1]
}

/* 
	方法二：DP + 贪心算法
	思路：
		通过 方法一 可知，我们在寻找最少跳跃次数过程中，把能够到达位置 nums[i] 的
		每一个可能的位置都进行了计算，而因为题目是需要寻找最少跳跃次数的，所以在寻找
		能否到达 nums[i] 的过程中，我们只需要从头寻找第一次能够跳到 nums[i] 的位置
		就行了，此后的位置就都不需要再计算了。
	时间复杂度：O(n^2)
		n 表示数组元素个数，因为对于每一个是否能到达的位置i只需要找一次。
		所以实际速度要比方法一快很多。
	空间复杂度：O(n)
		因为dp[i] 是基于 dp[0] ~ dp[i-1]的，所以空间复杂度无法优化。
*/
func jump2(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	// 以 dp[i] 表示从 nums[0] 到达 nums[i] 所需的最少跳跃次数
	dp := make([]int, n)
	dp[0] = 0
	for i := 1; i < n; i ++ {
		// 取第一个能跳到当前位置的点即可
        // 因为从距离越远的地方能够跳到，则需要的跳跃次数就越小
		idx := 0
        for idx < i && idx + nums[idx] < i {
            idx ++
		}
		// 跳出循环时 idx + nums[idx] >= i，即表示可以从 nums[idx] 处直接跳到 nums[i]
        dp[i] = dp[idx] + 1
	}
	return dp[n-1]
}

/* 
	方法三：贪心算法-寻找最远能到达的位置
	思路：
		我们遍历数组，查看每一个位置能够到达的最远位置，取最远位置的最大值，
		若遍历过程中抵达了之前所得到的最远位置，则说明在前面的一次跳跃中，
		最远只能到达这里，想要往后则需要再次跳跃，所以可以在此处更新跳跃次数，
		并清空能够到达的最远位置，重复上面的步骤。

		在具体的实现中，我们维护当前能够到达的最大下标位置，记为边界。
		我们从左到右遍历数组，到达边界时，更新边界并将跳跃次数增加 1。
		在遍历数组时，我们不访问最后一个元素，这是因为在访问最后一个元素之前，
		我们的边界一定大于等于最后一个位置，否则就无法跳到最后一个位置了。
		如果访问最后一个元素，在边界正好为最后一个位置的情况下，
		我们会增加一次「不必要的跳跃次数」，因此我们不必访问最后一个元素。

	时间复杂度：O(n)
		n 表示数组元素个数，我们只需要遍历一次。
	空间复杂度：O(1)
*/
func jump3(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	end := 0			// 每一跳按最大步数来跳跃时，已经到达的位置
	maxPosition := 0	// 当前跳可以跳到的最远位置
	step := 0			// 需要的跳跃次数
	// n - 1：不访问最后的位置
	for i := 0; i < n - 1; i ++ {
		// 取当前跳可以达到的最远位置
		maxPosition = max(maxPosition, i + nums[i])
		if i == end {
			// 更新已到达的位置
			end = maxPosition
			step ++
		}
	}
	return step
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

/* 
================== 9、分割回文串 II ==================
给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。
返回符合要求的最少分割次数。

示例:
输入: "aab"
输出: 1
解释: 进行一次分割就可将 s 分割成 ["aa","b"] 这样两个回文子串。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/palindrome-partitioning-ii
*/

/* 
	方法一：DP
	思路：
		步骤 1：思考状态
			状态就尝试定义成题目问的那样，看看状态转移方程是否容易得到。
			dp[i]：表示前缀子串 s[0:i] 分割成若干个回文子串所需要最小分割次数。
		步骤 2：思考状态转移方程
			思考的方向是：大问题的最优解怎么由小问题的最优解得到。
			即 dp[i] 如何与 dp[i - 1]、dp[i - 2]、...、dp[0] 建立联系。
			比较容易想到的是：如果 s[0:i] 本身就是一个回文串，那么不用分割，即 dp[i] = 0 ，
			这是首先可以判断的，否则就需要去遍历；
			接下来枚举可能分割的位置：即如果 s[0:i] 本身不是一个回文串，就尝试分割，枚举分割的边界 j。
			如果 s[j + 1, i] 不是回文串，尝试下一个分割边界。
			如果 s[j + 1, i] 是回文串，则 dp[i] 就是在 dp[j] 的基础上多一个分割。
			于是枚举 j 所有可能的位置，取所有 dp[j] 中最小的再加 1 ，就是 dp[i]。
			得到状态转移方程如下：
				dp[i] = min([dp[j] + 1 for j in range(i) if s[j + 1, i] 是回文])
		步骤 3：思考初始状态
			初始状态：单个字符一定是回文串，因此 dp[0] = 0。
		步骤 4：思考输出
			状态转移方程可以得到，并且状态就是题目问的，因此返回最后一个状态即可，即 dp[len - 1]。
		步骤 5：思考是否可以优化空间
			每一个状态值都与之前的状态值有关，因此不能优化空间。
	时间复杂度：O(n^2)
		n 为字符串的长度，我们需要用 O(n^2) 时间先来预处理每一个子串是不是回文，
		之后再用 O(n^2) 时间去寻找最小切割数，总的时间复杂度为O(2*n^2)，即O(n^2)
	空间复杂度：O(n^2)
		我们需要一个长度为 n 的二维数组去存储每一个子串是不是回文。
*/
func minCut(s string) int {
	n := len(s)
	if n < 2 {
		return 0
	}
	// 先计算每一个子串是不是回文，dp[i][j] 表示子串 s[i:j] 是不是回文
	dp := make([][]bool, n)
	for i := 0; i < n; i ++ {
		dp[i] = make([]bool, n)
	}
	palindromeInit(s, dp)
	// 用 f[i] 表示前缀子串 s[0:i] 分割成若干个回文子串所需要最小分割次数
	f := make([]int, n)
	f[0] = 0
	for i := 1; i < n; i ++ {
		// 假设s[0:i] 不是回文，则其需要的最大切割数为i
		f[i] = i
		// 子串本身是回文，则不需要分割
		if dp[0][i] {
			f[i] = 0
			continue
		}
		for j := 0; j < i; j ++ {
			// 这里 j 在前面
			if dp[j+1][i] {
				f[i] = min(f[i], f[j] + 1)
			}
		}
	}
	return f[n-1]
}
// 初始化每一个子串是不是回文
func palindromeInit(s string, dp [][]bool) {
	n := len(s)
	// sl 为子序列的长度
	for sl := 0; sl < n; sl ++ {
		for i := 0; i + sl < n; i ++ {
			j := i + sl
			if j - i == 0 {
				// 只有一个字符
				dp[i][j] = true
			} else if j - i == 1 && s[i] == s[j] {
				// 有两个字符
				dp[i][j] = true
			} else {
				// 三个以上的字符
				if dp[i+1][j-1] && s[i] == s[j] {
					dp[i][j] = true
				}
			}
		}
	}
}

/* 
================== 10、最长上升子序列 ==================
给定一个无序的整数数组，找到其中最长上升子序列的长度。

示例:
输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。

说明:
    可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。
    你算法的时间复杂度应该为 O(n2) 。

进阶: 你能将算法的时间复杂度降低到 O(nlogn) 吗?

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/longest-increasing-subsequence
*/
/* 
	方法一：初始DP
	思路：
		以 dp[i] 表示从0开始到i结尾的最长序列长度。
		初始状态：
			dp[0] = 0
		中间状态：
			dp[i] = max(dp[j]) + 1, a[j] < a[i]
			dp[i] 的最小值是1。
		终止状态：
			dp[n-1]
		结果：
			从 dp[0] ~ dp[n-1] 中找出最大值
		时间复杂度：O(n^2)
			n 表示数据元素的个数
		空间复杂度：O(n)
			每一个状态都依赖于前面的所有状态，所以不能优化。
*/
func lengthOfLIS(nums []int) int {
	n := len(nums)
    if n == 0 || n == 1 {
        return n
    }
    dp := make([]int, n)
	dp[0] = 1
	ans := dp[0]
    for i := 1; i < len(nums); i++ {
        dp[i] = 1
        for j := 0; j < i; j++ {
            if nums[j] < nums[i] {
                dp[i] = max(dp[i], dp[j] + 1)
            }
		}
		ans = max(ans, dp[i])
    }
    return ans
}

/* 
	方法二：动态规划 + 二分查找
	
*/

/* 
================== 11、最长【连续】上升子序列 ==================
给定一个无序的整数数组，找到其中最长连续上升子序列的长度。

示例:
输入: [10,9,2,5,3,7,101,18]
输出: 3 
解释: 最长的连续上升子序列是 [3,7,101]，它的长度是 3。

说明:
    可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。
    你算法的时间复杂度应该为 O(n2) 。

进阶: 你能将算法的时间复杂度降低到 O(n log n) 吗?

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/longest-increasing-subsequence
*/
/* 
	方法一：初始DP
	思路：
		使用 dp[i][j] 来表示 nums[i:j] 是不是连续上升序列。
		初始状态：
			只有一个元素时是连续上升序列，即：
				dp[i][j] = true, i == j
			有两个元素时，只要 nums[i] <= nums[j] 就是连续上升序列，即：
				dp[i][j] = (nums[i] <= nums[j]), j - i == 1
		中间状态：
			对于 nums[i:j] 是不是连续上升序列，只需要
			（nums[i] < nums[i+1] 且 nums[i+1:j] 是连续上升序列) 或 (nums[j] > nums[j-1] && nums[i:j-1] 是上升序列) 
			即：
				dp[i][j] = (nums[i] <= num[i+1] && dp[i+1][j]) || (dp[i][j-1] && nums[j] >= nums[j-1]),  0 < i < j < n
		终止状态：
			j == n - 1
		结果：
			找出 j-i 的最大值
		时间复杂度：O(n^2)
			n 表示数组元素的个数。
		空间复杂度：O(n^2)
*/
func lengthOfLIS2(nums []int) int {
	n := len(nums)
	dp := make([][]bool, n)
	for i := 0; i < n; i ++ {
		dp[i] = make([]bool, n)
	}
	ans := 0
	// sl 表示最长连续上升子序列的长度，0 表示只有一个元素
	for sl := 0; sl < n; sl ++ {
		for i := 0; i + sl < n; i ++ {
			j := i + sl
			if j - i == 0 {
				dp[i][j] = true
			} else if j - i == 1 && nums[i] <= nums[j] {
				dp[i][j] = true
			} else {
				if (nums[i] <= nums[i+1] && dp[i+1][j]) || (nums[j] >= nums[j-1] && dp[i][j-1]) {
					dp[i][j] = true
				}
			}
			if dp[i][j] && sl > ans {
				ans = sl
			}
		}
	}
	// sl 是从 0 开始的，最后要 + 1
	return ans + 1
}

/* 
================== 12、单词拆分 ==================
给定一个非空字符串 s 和一个包含非空单词的列表 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。
说明：
    拆分时可以重复使用字典中的单词。
    你可以假设字典中没有重复的单词。

示例 1：
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。

示例 2：
输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。

示例 3：
输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/word-break
*/
/* 
	方法一：动态规划
	思路：
		我们用 dp[i] 来表示前 i 个字符是否可以被拆分成字典中的单词，即 
			s[0..i−1] 是否能被空格拆分成若干个字典中出现的单词。
		初始状态：
			添加一个标记位：dp[0] = true，表示空串且合法。
		中间状态：
			dp[i] = d[j] && s[j,i] 是字典中的单词
		终止状态：
			因为我们添加了一个标记位，所以终止状态在 dp[n] 中
	时间复杂度：O(n^2)
		n 表示字符串的长度
	空间复杂度：O(n^2)
*/
func wordBreak(s string, wordDict []string) bool {
	n := len(s)
	if n == 0 {
		return false
	}
	// 使用 hash 表来缩短查找是不是字典中的单词的时间
	wordDictSet := make(map[string]bool, len(wordDict))
	for _, word := range wordDict {
		wordDictSet[word] = true
	}
	dp := make([]bool, n + 1)
	dp[0] = true
	for i := 1; i <= n; i ++ {
		for j := 0; j < i; j ++ {
			// 从 0 开始，即从字符最长的单词开始查找，找到一个符合条件就行
			if dp[j] && wordDictSet[s[j:i]] {
				dp[i] = true
				break
			}
		}
	}
	return dp[n]
}

/* 
================== 13、最长公共子序列 ==================
给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。
一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的
相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。
若这两个字符串没有公共子序列，则返回 0。

示例 1:
输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace"，它的长度为 3。

示例 2:
输入：text1 = "abc", text2 = "abc"
输出：3
解释：最长公共子序列是 "abc"，它的长度为 3。

示例 3:
输入：text1 = "abc", text2 = "def"
输出：0
解释：两个字符串没有公共子序列，返回 0。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/longest-common-subsequence
*/
/* 
    方法一：初始DP
    思路：
        我们可以基于两个字符串 s1 和 s2 来构建一个 dp 二维表，dp[i][j]表示
        s1 与 s2 的最长公共子序列长度（LCS），以 s1 = "ace", text2 = "abcde" 举例，
        表如下：
            s1\s2    0    1    2    3    4    5
                    ''    a    b    c    d    e
            0    ''    0    0    0    0    0    0
            1    a    0    1    1    1    1    1
            2    c    0    1    1    2    2    2
            3    e    0    1    1    2    2    3
        注：空串与任何字符串的最长公共子序列的长度都为0
        初始状态：
            两个空串的 LSC 为 0，即：
                dp[0][0] = 0
        中间状态：
            用两个指针遍历两个字符串，如果有 s1[i-1] == s2[j-1]，
            （-1 是因为 dp 记录了空串，所以 dp 的 i、j 对于字符串来说是大一位的）
            即意味着当前字符在 LCS 中，则：
                dp[i][j] = dp[i-1][j-1] + 1
            否则我们可以从：
                1、s1[i] 不在 LCS 中
                2、s2[j] 不在 LCS 中
                3、s1[i]、s2[j] 都不在 LCS 中
            这三种情况中找出最长的 LCS，但因为 情况3 无论如何都会小于 情况1 与 情况2,
            所以可以简写为：
                dp[i][j] = max(d[i-1][j], dp[i][j-1])
        终止状态：
            因为加入了空串，所以 LCS 在 dp[len(s1)][len(s2)] 中
    时间复杂度：O(m*n)
        m、n 分别表示字符串 s1、s2 的长度。
    空间复杂度：O(m*n)
        我们需要构建一个 m*n 的二维状态表。
*/
func longestCommonSubsequence(text1 string, text2 string) int {
    m, n := len(text1), len(text2)
    if m == 0 || n == 0 {
        return 0
    }
    // 因为加入了空串，所以长度 + 1
    dp := make([][]int, m + 1)
    for i := 0; i <= m; i ++ {
        dp[i] = make([]int, n + 1)
    }
	dp[0][0] = 0
	// 空串与任何字符串的 LCS 都是 0，即 dp[0][j] = 0, dp[i][0] = 0
	// 所以直接从 1 开始
    for i := 1; i <= m; i ++ {
        for j := 1; j <= n; j ++ {
            // -1 是因为 dp 记录了空串，所以 dp 的 i、j 对于字符串来说是大一位的
            if text1[i-1] == text2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            }
        }
    }
    return dp[m][n]
}
/* 
    方法二：DP + 空间优化
    思路：
        从方法一中我们可以看出，对于每一个状态 dp[i][j]，它只与 dp[i-1][j-1]、
        dp[i-1][j]、dp[i][j-1] 这三个前置状态有关，所以我们完全可以只用两个
        长度为 n + 1 的数组来完成整个状态转移过程，再借助于奇偶性来确定每一次该更新
        哪一个数组的状态就行了。
    时间复杂度：O(m*n)
        m、n 分别表示字符串 s1、s2 的长度。
    空间复杂度：O(2n)
        我们需要构建两个长度为 n + 1 的数组来完成状态转移。
    思考：
        能不能优化到 O(n) 呢？
        因为 dp[i][j] 与 dp[i-1][j-1]、dp[i-1][j]、dp[i][j-1]，如果我们只用一个
        数组来记录状态转移，那么 dp[i][j] 对应的就是 dp[j]，按 dp[i-1][j-1]、
        dp[i-1][j]、dp[i][j-1] 三个状态在 dp[j] 上的投影，就是 dp[j-1]、dp[j]、dp[j-1]，
        可以看出在修改 dp[j-1] 时会导致两个状态同时被影响，由此会影响到整个状态转移过程，
        所以不能优化到 O(n)。
*/
func longestCommonSubsequence2(text1 string, text2 string) int {
    m, n := len(text1), len(text2)
    if m == 0 || n == 0 {
        return 0
    }
    dp := make([][]int, 2)
    for i := 0; i < 2; i ++ {
        dp[i] = make([]int, n + 1)
	}
	cur, pre := 0, 0
	dp[0][0] = 0
	// 空串与任何字符串的 LCS 都是 0，即 dp[0][j] = 0, dp[i][0] = 0
	// 所以直接从 1 开始
    for i := 1; i <= m; i ++ {
        cur = i & 1    // 同 i % 2
        pre = 1 - cur
        for j := 1; j <= n; j ++ {
            // -1 是因为 dp 记录了空串，所以 dp 的 i、j 对于字符串来说是大一位的
            if text1[i-1] == text2[j-1] {
                dp[cur][j] = dp[pre][j-1] + 1
            } else {
                dp[cur][j] = max(dp[pre][j], dp[cur][j-1])
            }
        }
    }
    return dp[cur][n]
}

/* 
================== 14、编辑距离 ==================
给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。
你可以对一个单词进行如下三种操作：
    插入一个字符
    删除一个字符
    替换一个字符

示例 1：
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')

示例 2：
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/edit-distance
*/
/* 
	方法一：初始DP
	思路：
		我们可以对任意一个单词进行三种操作：
			插入一个字符；
			删除一个字符；
			替换一个字符。
		题目给定了两个单词，设为 A 和 B，这样我们就能够六种操作方法。
		但我们可以发现，如果我们有单词 A 和单词 B：
			对单词 A 删除一个字符和对单词 B 插入一个字符是等价的。例如当
			单词 A 为 doge，单词 B 为 dog 时，我们既可以删除单词 A 的最
			后一个字符 e，得到相同的 dog，也可以在单词 B 末尾添加一个字符 e，
			得到相同的 doge；
			同理，对单词 B 删除一个字符和对单词 A 插入一个字符也是等价的；
			对单词 A 替换一个字符和对单词 B 替换一个字符是等价的。例如当单
			词 A 为 bat，单词 B 为 cat 时，我们修改单词 A 的第一个字母 
			b -> c，和修改单词 B 的第一个字母 c -> b 是等价的。
		这样以来，本质不同的操作实际上只有三种：
			在单词 A 中插入一个字符；
			在单词 B 中插入一个字符；
			修改单词 A 的一个字符。
		这样以来，我们就可以把原问题转化为规模较小的子问题。我们用 A = horse，
		B = ros 作为例子，来看一看是如何把这个问题转化为规模较小的若干子问题的。
			在单词 A 中插入一个字符：如果我们知道 horse 到 ro 的编辑距离为 a，
			那么显然 horse 到 ros 的编辑距离不会超过 a + 1。这是因为我们可以
			在 a 次操作后将 horse 和 ro 变为相同的字符串，只需要额外的 1 次操作，
			在单词 A 的末尾添加字符 s，就能在 a + 1 次操作后将 horse 和 ro 变为
			相同的字符串；
			在单词 B 中插入一个字符：如果我们知道 hors 到 ros 的编辑距离为 b，
			那么显然 horse 到 ros 的编辑距离不会超过 b + 1，原因同上；
			修改单词 A 的一个字符：如果我们知道 hors 到 ro 的编辑距离为 c，
			那么显然 horse 到 ros 的编辑距离不会超过 c + 1，原因同上。
		那么从 horse 变成 ros 的编辑距离应该为 min(a + 1, b + 1, c + 1)。

		故本题与【最长公共子序列】的题目类似，
		以 dp[i][j] 表示字符串 s1 的前 i 个字符编辑为 s2 前 j 个字符所需的
		最少操作次数。我们用 s1、s2 组成一个 dp 状态表。
		表如下：
			s1\s2	0	1	2	3
					''	r	o	s
			0	''	0	1	2	3
			1	h	1	1	2	3
			2	o	2	2	1	2
			3	r	3	2	2	2
			4	s	4	3	3	2
			5	e	5	4	4	3
		解析：
			1、往右:
				表示 s2 比 s1 多了一个字符，此时要想s1、s2相同，则需要的操作是从 s2 中
				删除一个字符，或者是在 s1 中添加一个字符，其编辑距离为 1
			2、往下：
				表示在 s1 比 s2 多了一个字符，此时要想s1、s2相同，则需要的操作是从 s1 中
				删除一个字符，或者是在 s2 中添加一个字符，其编辑距离为 1
			3、往右下：
				表示在s1、s2字符数相同，但最后一个字符可能不一样， 此时要想s1、s2相同，则
				需要替换 s1 或者 s2 的最后一个字符使它们相同，即修改，其编辑距离为 1
		初始状态：
			我们记录空串，空串与空串相等，所以不需要编辑，即：
				dp[0][0] = 0
			空串与任何字符串对比所需的编辑次数是该字符串字符的个数，即：
				dp[0][j] = j
				dp[i][0] = i
		中间状态：
			对于 dp[i][j] 来说，
			如果 s1[i-1] == s2[j-1]（此处之所以 -1 是因为dp 中记录了空串），
				就不需要任何操作，即：
					dp[i][j] = dp[i-1][j-1], s[i-1] == s[j-1], i∈[1, len(s1)), j∈[1,len(s2))
			否则我们需要从以下几种情况中找出最小值，然后其操作次数 +1 ：
				1、曾：对 s1 的曾相当于对 s2 的减，即 dp[i][j-1]
				2、删：删除 s1 的一个字符，即 dp[i-1][j]
				3、改：相当于同时删除 s1、s2 的一个字符，即 dp[i-1][j-1]
				即：
					dp[i][j] = min(min(dp[i-1][j], dp[i][j-1]), dp[i-1][j-1]) + 1
		终止状态：
			我们需要遍历完两个字符串，所以结果保存在 dp[len(s1)][len(s2)] 中
		时间复杂度：O(m*n)
			m、n 分别表示 s1、s2 的长度
		空间复杂度：0(m*n)
*/
func minDistance(word1 string, word2 string) int {
	m, n := len(word1), len(word2)
	if m == 0 {
		return n
	}
	if n == 0 {
		return m
	}
	dp := make([][]int, m + 1)
	// 初始化
	for i := 0; i <= m; i ++ {
		dp[i] = make([]int, n + 1)
		dp[i][0] = i
	}
	for j := 1; j <= n; j ++ {
		dp[0][j] = j
	}
	for i := 1; i <= m; i ++ {
		for j := 1; j <= n; j ++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = min(min(dp[i][j-1], dp[i-1][j]), dp[i-1][j-1]) + 1
			}
		}
	}
	return dp[m][n]
}

/* 
	方法二：DP-空间优化
	思路：
		由方法一我们可知，对于每一个状态 dp[i][j]，它只与 dp[i][j-1]、dp[i-1][j]、
		dp[i-1][j-1] 三个前置状态有关，所以我们完全可以只用两个长度为 n 的数组来完成
		状态转移过程，由此可以将空间复杂度优化到 O(2n)
		终止状态：
			我们需要遍历完两个字符串，所以结果保存在 dp[len(s1)][len(s2)] 中
		时间复杂度：O(m*n)
			m、n 分别表示 s1、s2 的长度
		空间复杂度：0(2n)
*/
func minDistance2(word1 string, word2 string) int {
	m, n := len(word1), len(word2)
	if m == 0 {
		return n
	}
	if n == 0 {
		return m
	}
	dp := make([][]int, 2)
	for i := 0; i < 2; i ++ {
		dp[i] = make([]int, n + 1)
	}
	cur, pre := 0, 0
	for i := 0; i <= m; i ++ {
		cur = i & 1
		pre = 1 - cur
		for j := 0; j <= n; j ++ {
			// 把初始化放到这里
			if i == 0 {
				dp[cur][j] = j
				continue
			}
			if j == 0 {
				dp[cur][j] = i
				continue
			}
			if word1[i-1] == word2[j-1] {
				dp[cur][j] = dp[pre][j-1]
			} else {
				dp[cur][j] = min(min(dp[cur][j-1], dp[pre][j]), dp[pre][j-1]) + 1
			}
		}
	}
	return dp[cur][n]
}

// ================== 案列测试 ==================

// 2、测试最长连续序列
func longestConsecutiveTest() {
	nums := []int{1,2,0,1}
	res := longestConsecutive(nums)
	fmt.Println(res)
}

// 3、测试最小路径和
func minPathSumTest() {
	grid := [][]int{
		{1,3,1},
		{1,5,1},
		{4,2,1},
	}
	// res := minPathSum(grid)
	// res := minPathSum(grid)
	res := minPathSum3(grid)
	fmt.Println(res)
}

// 4、测试不同路径
func uniquePathsTest() {
	// res := uniquePaths(7,3)
	// res := uniquePaths2(3,2)
	res := uniquePaths3(7,3)
	fmt.Println(res)
}

// 5、测试不同路径 II
func uniquePathsWithObstaclesTest() {
	grid := [][]int{
		{0,0,0},
		{0,1,0},
		{0,0,0},
	}
	// res := uniquePathsWithObstacles(grid)
	res := uniquePathsWithObstacles2(grid)
	fmt.Println(res)
}

// 6、测试爬楼梯
func climbStairsTest() {
	n := 3
	// res := climbStairs(n)
	res := climbStairs2(n)
	fmt.Println(res)
}

// 7、测试跳跃游戏
func canJumpTest() {
	nums := []int{3,2,1,0,4}
	res := canJump(nums)
	fmt.Println(res)
}

// 8、测试跳跃游戏 II
func jumpTest() {
	nums := []int{2,3,1,1,4}
	// res := jump(nums)
	// res := jump2(nums)
	res := jump3(nums)
	fmt.Println(res)
}

// 9、测试分割回文串 II
func minCutTest() {
	s := "aab"
	res := minCut(s)
	fmt.Println(res)
}

// 10、测试最长上升子序列
func lengthOfLISTest() {
	nums := []int{10,9,2,5,3,7,101,18}
	res := lengthOfLIS(nums)
	fmt.Println(res)
}

// 11、测试最长连续上升子序列
func lengthOfLIS2Test() {
	nums := []int{10,9,2,5,3,7,101,18}
	res := lengthOfLIS2(nums)
	fmt.Println(res)
}

// 12、测试单词拆分
func wordBreakTest() {
	s := "leetcode"
	wordDict := []string{"leet", "code"}
	res := wordBreak(s, wordDict)
	fmt.Println(res)
}

// 13、测试最长公共子序列
func longestCommonSubsequenceTest() {
	s1, s2 := "abcde", "ace"
	// res := longestCommonSubsequence(s1, s2)
	res := longestCommonSubsequence2(s1, s2)
	fmt.Println(res)
}

func main() {
	// longestConsecutiveTest()
	// minPathSumTest()
	// uniquePathsTest()
	// uniquePathsWithObstaclesTest()
	// climbStairsTest()
	// canJumpTest()
	// jumpTest()
	// minCutTest()
	// lengthOfLISTest()
	// lengthOfLIS2Test()
	// wordBreakTest()
	longestCommonSubsequenceTest()
}