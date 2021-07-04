package main

/*
============== 剑指 Offer 04. 二维数组中的查找 ==============
在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按
照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一
个整数，判断数组中是否含有该整数。

示例:
现有矩阵 matrix 如下：
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
给定 target = 5，返回 true。
给定 target = 20，返回 false。

限制：
0 <= n <= 1000
0 <= m <= 1000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof
*/
/*
	方法一：行二分查找
	思路：
		对每一行做二分查找
	时间复杂度：O(m*logn)
	空间复杂度：O(1)
*/
func findNumberIn2DArray(matrix [][]int, target int) bool {
	m := len(matrix)
	if m == 0 {
		return false
	}
	for i := 0; i < m; i++ {
		if binarySearch(matrix[i], target) {
			return true
		}
	}
	return false
}
func binarySearch(nums []int, target int) bool {
	n := len(nums)
	if n == 0 {
		return false
	}
	// 使用模板三
	L, R := 0, n-1
	for L+1 < R {
		mid := L + ((R - L) >> 1)
		if target < nums[mid] {
			R = mid
		} else {
			L = mid
		}
	}
	return nums[L] == target || nums[R] == target
}

/*
	方法二：二维数组中的查找
	思路：
		首先选取数组中右上角的数字 nums[i][j]，此时 nums[i][j] 在现有数组中
		是第 i 行的最大值和第 j 列的最小值。
			如果 nums[i][j] == target，说明找到目标，返回 true；
			如果 nums[i][j] > target，说明 target 在 j 列的左边，j --
			如果 nums[i][j] < target，说明 target 在 i 行的下边，i ++
		初始条件：i = 0, j = n - 1
		重复上述查找操作，直至找到目标或是遍历完成。
		注：
			也可以从左下角开始，但不能从左上角和右下角开始。

			此方法可以把二维数组看做类似一颗二叉搜索树，右上角为 root，左上
			为 left，右下为 right。
	时间复杂度：O(n+m)。
		访问到的下标的行最多增加 n 次，列最多减少 m 次，因此循环体最多执
		行 n + m 次。
	空间复杂度：O(1)
*/
func findNumberIn2DArray2(matrix [][]int, target int) bool {
	m := len(matrix)
	if m == 0 {
		return false
	}
	n := len(matrix[0])
	if n == 0 {
		return false
	}
	i, j := 0, n-1
	for i < m && j >= 0 {
		if matrix[i][j] == target {
			return true
		}
		if matrix[i][j] > target {
			j--
		} else {
			i++
		}
	}
	return false
}
