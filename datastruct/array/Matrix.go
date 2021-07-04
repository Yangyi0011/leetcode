package array

/* 
	二维数组（矩阵）
*/

/* 
========================== 1、旋转矩阵 ==========================
给你一幅由 N × N 矩阵表示的图像，其中每个像素的大小
为 4 字节。请你设计一种算法，将图像旋转 90 度。

不占用额外内存空间能否做到？ 

示例 1:
给定 matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],
原地旋转输入矩阵，使其变为:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]

示例 2:
给定 matrix =
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 
原地旋转输入矩阵，使其变为:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/clpgd/
*/
/* 
	方法一：原地旋转
	思路：
		我们用 matrix[i][j] 表示原矩阵的一个元素，N 表示矩阵的维数，
		观察一组数据的旋转变化情况后我们发现，对于每一个元素 matrix[i][j]
		的旋转，旋转后都会到达对应的 matrix[j][N - 1 - i] 位置，如此有：
			tmp = matrix[i][j]
			matrix[i][j] = matrix[N-1-j][i]
			matrix[N-1-j][i] = matrix[N-1-i][N-1-j]
			matrix[N-1-i][N-1-j] = matrix[j][N - 1 - i]
			matrix[j][N-1-i] = tmp
		根据旋转中元素的变化过程我们可以知道每一个元素旋转后该存入的具体位置，
		由此实现原地交换。当我们知道了如何原地旋转矩阵之后，还有一个重要的问题在于：
		我们应该枚举哪些位置 (i,j) 进行上述的原地交换操作呢？
		由于每一次原地交换四个位置，因此：
			当 n 为偶数时，我们需要枚举 n^2/4=(n/2)∗(n/2) 个位置，矩阵的左上角符合我们的要求。
			例如当 n=4 时，下面第一个矩阵中 ∗ 所在就是我们需要枚举的位置，
			每一个 ∗ 完成了矩阵中四个不同位置的交换操作：
				**..              ..**              ....              ....
				**..   =下一项=>   ..**   =下一项=>   ....   =下一项=>   ....
				....              ....              ..**              **..
				....              ....              ..**              **..
			保证了不重复、不遗漏；
			当 n 为奇数时，由于中心的位置经过旋转后位置不变，我们需要枚举 (n^2−1)/4=((n−1)/2)∗((n+1)/2) 个位置，
			同样可以使用矩阵左上角对应大小的子矩阵。例如当 n=5 时，下面第一个矩阵中 ∗ 所在就是我们需要枚举的位置，
			每一个 ∗ 完成了矩阵中四个不同位置的交换操作：
				***..              ...**              .....              .....
				***..              ...**              .....              .....
				..x..   =下一项=>   ..x**   =下一项=>   ..x..   =下一项=>   **x..
				.....              .....              ..***              **...
				.....              .....              ..***              **...
			同样保证了不重复、不遗漏。
		综上所述，我们只需要枚举矩阵左上角高为 ⌊n/2⌋，宽为 ⌊(n+1)/2⌋ 的子矩阵即可。
	时间复杂度：O(N^2)
		其中 N 是 matrix 的边长。对于每一次翻转操作，我们都需要枚举矩阵中一半的元素。
	空间复杂度：O(1)
		为原地翻转得到的原地旋转。
*/
func rotate(matrix [][]int)  {
	N := len(matrix)
	for i := 0; i < (N >> 1); i ++ {
		for j := 0; j < ((N + 1) >> 1); j ++ {
			tmp := matrix[i][j]
			matrix[i][j] = matrix[N-1-j][i]
			matrix[N-1-j][i] = matrix[N-1-i][N-1-j]
			matrix[N-1-i][N-1-j] = matrix[j][N - 1 - i]
			matrix[j][N-1-i] = tmp
		}
	}
}

/* 
========================== 2、零矩阵 ==========================
编写一种算法，若M × N矩阵中某个元素为0，则将其所在的行与列清零。

示例 1：
输入：
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]
输出：
[
  [1,0,1],
  [0,0,0],
  [1,0,1]
]

示例 2：
输入：
[
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]
输出：
[
  [0,0,0,0],
  [0,4,5,0],
  [0,3,1,0]
]

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/ciekh/
*/
/* 
	方法一：零元素分解投影
	思路：
		依题意可知，若M × N矩阵中某个元素为0，则将其所在的行与列清零。
		即如果 matrix[i][j] = 0，则 matrix[0~M][j] = 0，matrix[i][0~N] = 0，
		我们可以预先记录第一行和第一列是否包含0，然后遍历矩阵，把其他
		位置的0元素的清零操作分解为行、列两个方向投影到第一行、列中，把投影
		位置置为 0，然后我们再次从matrix[1,1]开始遍历，根据第一行/列是否是0
		来对当前元素做清零操作，最后再根据第一行、第一列是否包含0来处理第一行、
		第一列列的清零操作即可，
		这样我们就可以避免多0元素的重复操作问题和置0前后难以记录的问题了。
		图示：
			1 2 3 4					1 0 3 4					1 0 0 4
			2 0 2 4	--投影[1,1]的0-> 0 0 2 4				 0 0 2 4
			1 4 0 3					1 4 0 3	--投影[2,2]的0-> 0 4 0 3
			4 3 2 1					4 3 2 1					4 3 2 1
			对 0行、0列做列、行清零处理：
			投影前：		投影后：		清零：
				1 2 3 4			1 0 0 4		1 0 0 4	
				2 0 2 4			0 0 2 4		0 0 0 0
				1 4 0 3			0 4 0 3		0 0 0 0
				4 3 2 1			4 3 2 1		4 0 0 1
	时间复杂度：O(n)
		n 是矩阵元素个数，我们需要遍历两次矩阵。
	空间复杂度：O(1)
*/
func setZeroes(matrix [][]int)  {
	// 获取行数
	m := len(matrix)
	if m == 0 {
		return
	}
	// 获取列数
	n := len(matrix[0])
	if n == 0 {
		return
	}
	// 记录第一行和第一列的元素是否包含 0
	firstRowHasZero := false
	firstColumnHasZero := false
	for i := 0; i < n; i ++ {
		if matrix[0][i] == 0 {
			firstRowHasZero = true
			break
		}
	}
	for i := 0; i < m; i ++ {
		if matrix[i][0] == 0 {
			firstColumnHasZero = true
			break
		}
	}
	// 从 matrix[1][1] 开始处理行列的分解投影
	for i := 1; i < m; i ++ {
		for j := 1; j < n; j ++ {
			if matrix[i][j] == 0 {
				// 投影到第一行
				matrix[0][j] = 0
				// 投影到第一列
				matrix[i][0] = 0
			}
		}
	}
	// 根据第一行、第一列的值来对当前位置的元素置0
	for i := 1; i < m; i ++ {
		for j := 1; j < n; j ++ {
			if matrix[0][j] == 0 || matrix[i][0] == 0 {
				matrix[i][j] = 0
			}
		}
	}
	// 第一行的清零操作
	if firstRowHasZero {
		for i := 0; i < n; i ++ {
			matrix[0][i] = 0
		}
	}
	// 第一列的清零操作
	if firstColumnHasZero {
		for i := 0; i < m; i ++ {
			matrix[i][0] = 0
		}
	}
}

/* 
	方法二：存储包含0的行数和列数
	思路：
		把包含0的行数和列数都记录下来，再遍历这些行和列把元素设为0
	时间复杂度：O(N)
		N是矩阵元素个数，我们至少需要扫描2次矩阵
	空间复杂度：O(N)
		N是矩阵元素个数，我们需要用一个二维数组来存储值为0的元素所对应的行和列，
		最坏情况下所有元素都为0
*/
func setZeroes(matrix [][]int)  {
	m := len(matrix)
	if m == 0 {
		return
	}
	n := len(matrix[0])
	if n == 0 {
		return
	}
	// 用以存储0元素的横纵坐标
	arr := make([][]int, 0)
	for i := 0; i < m; i ++ {
		for j := 0; j < n; j ++ {
			if matrix[i][j] == 0 {
				arr = append(arr, []int{i, j})
			}
		}
	}
	for _, v := range arr {
		row, col := v[0], v[1]
		//行置0
		for j := 0; j < n; j ++ {
			matrix[row][j] = 0
		}
		//列置0
		for i := 0; i < m; i ++ {
			matrix[i][col] = 0
		}
	}
}

/* 
========================== 3、对角线遍历 ==========================
给定一个含有 M x N 个元素的矩阵（M 行，N 列），请以对角线遍历的顺序返回这个矩阵中的所有元素，对角线遍历如下图所示。

示例:
输入:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]

输出:  [1,2,4,7,5,3,6,8,9]
解释:  _   _
	 / / / /
	1 2 3 /
   / / / /
	4 5 6 /->
   / / / /
  / 7 8 9
 /_/ /_/

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cuxq3/
*/
/* 
	思路：
		1、每一趟对角线中元素的坐标（x, y）相加的和是递增的。
				第一趟：1 的坐标(0, 0)。x + y == 0。
				第二趟：2 的坐标(1, 0)，4 的坐标(0, 1)。x + y == 1。
				第三趟：7 的坐标(0, 2), 5 的坐标(1, 1)，3 的坐标(2, 0)。第三趟 x + y == 2。
				第四趟：……
		2、每一趟都是 x 或 y 其中一个从大到小（每次-1），另一个从小到大（每次+1）。
				第二趟：2 的坐标(0, 1)，4 的坐标(1, 0)。x 每次-1，y 每次+1。
				第三趟：7 的坐标(2, 0) 5 的坐标(1, 1)，3 的坐标(0, 2)。x 每次 +1，y 每次 -1。
		3、确定初始值。例如这一趟是 x 从大到小， x 尽量取最大，当初始值超过 x 的上限时，
			不足的部分加到 y 上面。
				第二趟：2 的坐标(0, 1)，4 的坐标(1, 0)。x + y == 1，x 初始值取 1，y 取 0。
				第四趟：6 的坐标(1, 2)，8 的坐标(2, 1)。x + y == 3，x 初始值取 2，剩下的加到 y上，y 取 1。
		4、确定结束值。例如这一趟是 x 从大到小，这一趟结束的判断是， x 减到 0 或者 y 加到上限。
				第二趟：2 的坐标(0, 1)，4 的坐标(1, 0)。x 减到 0 为止。
				第四趟：6 的坐标(1, 2)，8 的坐标(2, 1)。x 虽然才减到 1，但是 y 已经加到上限了。
		5、这一趟是 x 从大到小，那么下一趟是 y 从大到小，循环进行。 
			并且方向相反时，逻辑处理是一样的，除了x，y和他们各自的上限值是相反的。
				x 从大到小，第二趟：2 的坐标(0, 1)，4 的坐标(1, 0)。x + y == 1，
					x 初始值取 1，y 取 0。结束值 x 减到 0 为止。
				x 从小到大，第三趟：7 的坐标(2, 0)，5 的坐标(1, 1)，3 的坐标(0, 2)。
					x + y == 2，y 初始值取 2，x 取 0。结束值 y 减到 0 为止。
*/
func findDiagonalOrder(matrix [][]int) []int {
	res := make([]int, 0)
	m := len(matrix)
	if m == 0 {
		return res
	}
	n := len(matrix[0])
	if n == 0 {
		return res
	}

	// 标识方向，以确定上限的转换
	flag := true
	for i := 0; i < m + n - 1; i ++ {
		// 根据方向确定上限
		var pm, pn int
		if flag {
			pm, pn = m, n
		} else {
			pm, pn = n, m
		}

		var x, y int
		if i < pm {
			x = i
		} else {
			// 不能超过最大值
			x = pm - 1
		}
		y = i - x
		for x >= 0 && y < pn {
			if flag {
				res = append(res, matrix[x][y])
			} else {
				res = append(res, matrix[y][x])
			}
			x --
			y ++
		}
		flag = !flag
	}
	return res
}

/* 
========================== 4、杨辉三角 ==========================
给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。
			1
		  1   1
		1   2   1
	  1   3   3   1
	1   4   6   4   1
	在杨辉三角中，每个数是它左上方和右上方的数的和

输入: 5
输出:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cuj3m/
*/
/* 
	方法一：动态规划
	思路：
		把杨辉三角的元素左对齐，这样就得到了一个二维数组，然后再根据杨辉三角
		每一个元素的变化特性去生成杨辉三角就行了：
			[
			[1],
			[1,1],
			[1,2,1],
			[1,3,3,1],
			[1,4,6,4,1]
			]
		由杨辉三角的特性可知：每一行的第一个和最后一个元素都是1，其余元素是
		左上方元素与正上方元素的和，我们以 dp[i][j] 表示杨辉三角第i行第j列的
		元素的值，i、j 从0 开始，由此得出状态转移方程：
			dp[i][j] = 1, j == 0 | j == i
			dp[i][j] = dp[i-1][j-1] + dp[i-1][j], 0 < j < i
	时间复杂度：O(n^2)
	空间复杂度：O(n^2)
*/
func generate(numRows int) [][]int {
	if numRows == 0 {
		return [][]int{}
	}
	dp := make([][]int, numRows)
	for i := 0; i < numRows; i ++ {
		dp[i] = make([]int, i + 1)
		// 第一个元素为1
		dp[i][0] = 1
		for j := 1; j < i; j ++ {
			// 中间元素为左上方元素和正上方元素的和
			dp[i][j] = dp[i - 1][j-1] + dp[i - 1][j]
		}
		// 最后一个元素为1
		dp[i][i] = 1
	}
	return dp
}

/* 
========================== 5、杨辉三角 II ==========================
给定一个非负索引 k，其中 k ≤ 33，返回杨辉三角的第 k 行，k 从 0 开始。

示例:
输入: 3
输出: [1,3,3,1]

进阶：
你可以优化你的算法到 O(k) 空间复杂度吗？
*/
/* 
	方法一：动态规划-空间优化
	思路：
		由杨辉三角的生成过程我们已经得出它的状态转移方程为：
			dp[i][j] = 1, j == 0 | j == i
			dp[i][j] = dp[i-1][j-1] + dp[i-1][j], 0 < j < i
		即我们只需要知道当前行的数据就可以推出下一行的数据，所以我们完全可以
		只用两个长度为 rowIndex + 1 的数组，再配合奇偶数来记录动态转移过程，
		如此空间复杂度就能从 O(n^2) 优化到 O(2n)，即O(n)
	时间复杂度：O(n^2)
	空间复杂度：O(n)
		我们两个数组交替变化来完成状态转移过程。
*/
func getRow(rowIndex int) []int {
	if rowIndex < 0 {
		return []int{}
	}
	// 初始化 dp 数组
	dp := make([][]int, 2)
	for i := 0; i < 2; i ++ {
		dp[i] = make([]int, rowIndex + 1)
	}
	// 使用 pre、cur 来确定上一行和当前行
	pre, cur := 0, 0
	// rowIndex 是从 0 开始的
	for i := 0; i <= rowIndex; i ++ {
		cur = i & 1		// 等价于 i % 2
		pre = 1 - cur
		dp[cur][0] = 1
		for j := 1; j < i; j ++ {
			dp[cur][j] = dp[pre][j-1] + dp[pre][j]
		}
		dp[cur][i] = 1
	}
	// 最后处理的当前行即为目标行
	return dp[cur]
}

/* 
	方法二：动态规划-空间优化-续
	思路：
		经过方法一我们已经把空间复杂度优化到 O(2n) 了，那么我们能不能只用一个
		数组来完成整个状态转移过程呢？
			dp[i][j] = 1, j == 0 | j == i
			dp[i][j] = dp[i-1][j-1] + dp[i-1][j], 0 < j < i
		观察上面的状态转移方程，我们发现在一行当中，状态是按从左到右的顺序发生
		变化的，如果我们只用一个数组来记录当前行的状态和推导出下一行的状态，那么
		我们需要从右到左更新数组以得出下一行的状态，这样才不会因为下一行的数据
		覆盖掉当前行的数据而破坏后续的状态转移过程。
	时间复杂度：O(n^2)
	空间复杂度：O(n)
		我们只需要一个数组来完成状态转移过程。
*/
func getRow(rowIndex int) []int {
	if rowIndex < 0 {
		return []int{}
	}
	dp := make([]int, rowIndex + 1)
	for i := 0; i <= rowIndex; i ++ {
		// 从右往左更新数组
		dp[i] = 1
		for j := i - 1; j > 0; j -- {
			dp[j] = dp[j-1] + dp[j]
		}
		dp[0] = 1
	}
	return dp
}

