package main

import (
	"fmt"
	"math"
)

/* 
===================== 1、二分查找 =====================
给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，
写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

示例 1:
输入: nums = [-1,0,3,5,9,12], target = 9
输出: 4
解释: 9 出现在 nums 中并且下标为 4

示例 2:
输入: nums = [-1,0,3,5,9,12], target = 2
输出: -1
解释: 2 不存在 nums 中因此返回 -1

提示：
    你可以假设 nums 中的所有元素是不重复的。
    n 将在 [1, 10000]之间。
    nums 的每个元素都将在 [-9999, 9999]之间。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/binary-search
*/
/* 
	二分查找：
		二分查找一般由三个主要部分组成：
			1、预处理：
				如果集合未排序，则进行排序。
			2、二分查找：
				使用循环或递归在每次比较后将查找空间划分为两半。
			3、后处理：
				在剩余空间中确定可行的候选者。
		常用的有三个模板：
			1、(left <= right)
				每次循环查找空间只有一个元素（nums[mid]），
				循环结束（left == right + 1）即查找完成，不需要做后续处理。
			2、(left < right)
				每次循环查找空间有两个元素（nums[mid]、nums[mid+1]），
				循环结束时（left == right）会剩余一个元素 num[left]，需要做后续处理。
			3、(left + 1 < right)
				每次循环查找空间有三个元素（nums[mid - 1]、nums[mid]、nums[mid+1]），
				循环结束时（left + 1 == right）会剩余两个元素 nums[left]和nums[right]，需要做后续处理。
	时间复杂度：O(log n)
		二分查找是通过对查找空间中间的值应用一个条件来操作的，并因此将查找空间折半，
		在更糟糕的情况下，我们将不得不进行 O(log n) 次比较，其中 n 是集合中元素的数目。
	空间复杂度：O(1)
		我们只需要3个变量的额外空间。
*/
func search(nums []int, target int) int {
	return binarySearch_2(nums, target)
}

/* 
	二分查找模板一：(left <= right)
		关键属性：
			二分查找的最基础和最基本的形式。
			查找条件可以在不与元素的两侧进行比较的情况下确定（或使用它周围的特定元素）。
			不需要后处理，因为每一步中，你都在检查是否找到了元素。如果到达末尾，则知道未找到该元素。
		区分语法：
			初始条件：left = 0, right = length-1
			终止条件：left > right
			向左查找：right = mid-1
			向右查找：left = mid+1
			后续处理：无
*/
func binarySearch_1(nums []int, target int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	L, R := 0, n - 1
	for L <= R {
		// 此处不能用(L+R)/2，因为加完后有溢出的风险
		// 同 L + (R - L) / 2，加括号是因为算数运算符的优先级大于位运算符
		mid := L + ((R - L) >> 1)
		if nums[mid] == target {
			return mid
		}
		if target > nums[mid] {
			L = mid + 1
		} else {
			R = mid - 1
		}
	}
	// L == R + 1 时循环停止，此时查找完成，无需后续处理
	return -1
}

/* 
	二分查找模板二：(left < right)
		一种实现二分查找的高级方法。
		查找条件需要访问元素的 直接右邻居。
		使用元素的右邻居来确定是否满足条件，并决定是向左还是向右。
		保证查找空间在每一步中至少有 2 个元素。
		需要进行后续处理。 
		当你剩下 1 个元素时，循环 / 递归结束。 需要评估剩余元素是否符合条件。
	区分语法：
		初始条件：left = 0, right = n
		终止条件：left = right
		向左查找：right = mid
		向右查找：left = mid + 1
		后续处理：
			if left != n && nums[left] == target {
				return L
			}
*/
func binarySearch_2(nums []int, target int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	L, R := 0, n
	for L < R {
		// 此处不能用(L+R)/2，因为加完后有溢出的风险
		// 同 L + (R - L) / 2，加括号是因为算数运算符的优先级大于位运算符
		mid := L + ((R - L) >> 1)
		if target > nums[mid] {
			L = mid + 1
		} else {
			R = mid
		}
	}
	// L == R 时循环停止，此时会剩余一个元素需要判断
	if L != n && nums[L] == target {
		return L
	}
	return -1
}

/* 
	二分查找模板三：(left + 1 < right)
		实现二分查找的另一种方法。
		搜索条件需要访问元素的直接左、右邻居。
		使用元素的邻居来确定它是向右还是向左。
		保证查找空间在每个步骤中至少有 3 个元素。
		需要进行后处理。 当剩下 2 个元素时，循环 / 递归结束。 需要评估其余元素是否符合条件。
*/
func binarySearch_3(nums []int, target int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	L, R := 0, n - 1
	for L + 1 < R {
		// 此处不能用(L+R)/2，因为加完后有溢出的风险
		mid := L + (R - L) >> 1
		if target > nums[mid] {
			L = mid
		} else {
			R = mid
		}
	}
	// L + 1 == R 时循环停止，此时会剩余两个元素需要判断
	if nums[L] == target {
		return L
	}
	if nums[R] == target {
		return R
	}
	return -1
}

/* 
===================== 2、x 的平方根 =====================
实现 int sqrt(int x) 函数。
计算并返回 x 的平方根，其中 x 是非负整数。
由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

示例 1:
输入: 4
输出: 2

示例 2:
输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xe9cog/
*/

/* 
	方法一：袖珍计算器算法
		算法与思路：
			袖珍计算器算法」是一种用指数函数 exp⁡ 和对数函数 ln⁡ 代替平方根函数的方法。
			我们通过有限的可以使用的数学函数，得到我们想要计算的结果。
			我们将 根号x 写成幂的形式 x^(1/2)，再使用自然对数 e 进行换底，即可得到
				根号x = x^(1/2)= [e^(ln⁡ x)]^(1/2) = e^[(1/2)*(ln⁡ x)]
			这样我们就可以得到 根号x 的值了。

		注意：
			由于计算机无法存储浮点数的精确值（浮点数的存储方法可以参考 IEEE 754，这里不再赘述），
			而指数函数和对数函数的参数和返回值均为浮点数，因此运算过程中会存在误差。例如当 x=2147395600 时，
			e^[(1/2)*(ln⁡ x)] 的计算结果与正确值 46340 相差 10^−11，这样在对结果取整数部分时，
			会得到 46339 这个错误的结果。
			因此在得到结果的整数部分 ans 后，我们应当找出 ans 与 ans+1 中哪一个是真正的答案。
		时间复杂度：O(1)
			由于内置的 exp 函数与 log 函数一般都很快，我们在这里将其复杂度视为 O(1)
		空间复杂度：O(1)
*/
func mySqrt(x int) int {
    if x == 0 {
        return 0
    }
    ans := int(math.Exp(0.5 * math.Log(float64(x))))
    if (ans + 1) * (ans + 1) <= x {
        return ans + 1
    }
    return ans
}

/* 
	方法二：二分查找
		思路与算法：
			由于 x 平方根的整数部分 ans 是满足 k^2 ≤ x 的最大 k 值，因此我们可以对 k 进行二分查找，从而得到答案。
			二分查找的下界为 0，上界可以粗略地设定为 x。在二分查找的每一步中，
			我们只需要比较中间元素 mid 的平方与 x 的大小关系，并通过比较的结果调整上下界的范围。
			由于我们所有的运算都是整数运算，不会存在误差，因此在得到最终的答案 ans 后，也就不需要再去尝试 ans+1 了。
		时间复杂度：O(log x)
			即为二分查找需要的次数。
		空间复杂度：O(1)
*/
func mySqrt2(x int) int {
	if x == 0 {
		return 0
	}
	L, R := 0, x
	ans := -1
	for L <= R {
		mid := L + (R - L ) >> 1
		if mid * mid <= x {
			ans = mid
			L = mid + 1	
		} else {
			R = mid - 1
		}
	}
	return ans
}

/* 
===================== 3、猜数字大小 =====================
猜数字游戏的规则如下：
    每轮游戏，我都会从 1 到 n 随机选择一个数字。 请你猜选出的是哪个数字。
    如果你猜错了，我会告诉你，你猜测的数字比我选出的数字是大了还是小了。
你可以通过调用一个预先定义好的接口 int guess(int num) 来获取猜测结果，返回值一共有 3 种可能的情况（-1，1 或 0）：
    -1：我选出的数字比你猜的数字小 pick < num
    1：我选出的数字比你猜的数字大 pick > num
    0：我选出的数字和你猜的数字一样。恭喜！你猜对了！pick == num

示例 1：
输入：n = 10, pick = 6
输出：6

示例 2：
输入：n = 1, pick = 1
输出：1

示例 3：
输入：n = 2, pick = 1
输出：1

示例 4：
输入：n = 2, pick = 2
输出：2

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xee4ev/
*/
// func guessNumber(n int) int {
// 	L, R := 1, n
// 	for L <= R {
// 		mid := L + (R - L) >> 1
// 		if guess(mid) == 0 {
// 			return mid
// 		}
// 		if guess(mid) == 1 {
// 			L = mid + 1
// 		} else {
// 			R = mid - 1
// 		}
// 	}
// 	return 0
// }

/* 
===================== 4、搜索旋转排序数组 =====================
给你一个整数数组 nums ，和一个整数 target 。
该整数数组原本是按升序排列，但输入时在预先未知的某个点上进行了旋转。
（例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] ）。
请你在数组中搜索 target ，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。

示例 1：
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4

示例 2：
输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1

示例 3：
输入：nums = [1], target = 0
输出：-1

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xeog5j/
*/

/* 
	算法与思路：
		虽然数组本身不是有序的，但是我们发现从中间分割数组的时候，，一定有一部分的数组是有序的。
		拿示例来看，我们从 6 这个位置分开以后数组变成了 [4, 5, 6] 和 [7, 0, 1, 2] 两个部分，
		其中左边 [4, 5, 6] 这个部分的数组是有序的，其他也是如此。

		这启示我们可以在常规二分搜索的时候查看当前 mid 为分割位置分割出来的两个部分 [l, mid] 
		和 [mid + 1, r] 哪个部分是有序的，并根据有序的那个部分确定我们该如何改变二分搜索的上下界，
		因为我们能够根据有序的那部分判断出 target 在不在这个部分：
			如果 [l, mid - 1] 是有序数组，且 target 的大小满足 [nums[l],nums[mid])，
				则我们应该将搜索范围缩小至 [l, mid - 1]，否则在 [mid + 1, r] 中寻找。
			如果 [mid, r] 是有序数组，且 target 的大小满足 (nums[mid+1],nums[r]]，
				则我们应该将搜索范围缩小至 [mid + 1, r]，否则在 [l, mid - 1] 中寻找。
*/
func search4(nums []int, target int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	if n == 1 {
		if nums[0] == target {
			return 0
		} else {
			return -1
		}
	}
	L, R := 0, n - 1
	for L + 1 < R {
		mid := L + (R - L) >> 1
		// 判断左边是否有序，左边有序则选择左边，否则选择右边
		if nums[0] <= nums[mid] {
			// 左边有序时，看看目标是否在左边，在则选择左边，否则选择右边
			if nums[0] <= target && target < nums[mid] {
				R = mid
			} else {
				L = mid
			}
		} else {
			// 左边无序时右边一定是有序的
			// 右边有序时，看看目标是否在右边，在则选择右边，否则选择左边
			if nums[mid] < target && target <= nums[n - 1] {
				L = mid
			} else {
				R = mid
			}
		}
	}
	if nums[L] == target {
		return L
	}
	if nums[R] == target {
		return R
	}
	return -1
}

/* 
===================== 5、第一个错误的版本 =====================
你是产品经理，目前正在带领一个团队开发新的产品。不幸的是，你的产品的最新版本没有通过质量检测。
由于每个版本都是基于之前的版本开发的，所以错误的版本之后的所有版本都是错的。
假设你有 n 个版本 [1, 2, ..., n]，你想找出导致之后所有版本出错的第一个错误的版本。
你可以通过调用 bool isBadVersion(version) 接口来判断版本号 version 是否在单元测试中出错。
实现一个函数来查找第一个错误的版本。你应该尽量减少对调用 API 的次数。

示例:
	给定 n = 5，并且 version = 4 是第一个错误的版本。
	调用 isBadVersion(3) -> false
	调用 isBadVersion(5) -> true
	调用 isBadVersion(4) -> true
所以，4 是第一个错误的版本。 

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xepthr/
*/
/* 
	算法：二分查找模板三
	思路：
		遇到错误版本就往左找，尝试找到第一个
	时间复杂度：O(log n)
	空间复杂度：O(1)
*/
// func firstBadVersion(n int) int {
// 	L, R := 1, n
// 	for L + 1 < R {
// 		mid := L + (R - L) >> 1
// 		// 是错误版本，需要往左找
// 		if isBadVersion(mid) {
// 			R = mid
// 		} else {
// 			L = mid
// 		}
// 	}
// 	if isBadVersion(L) {
// 		return L
// 	}
// 	return R
// }

/* 
===================== 6、搜索区间 =====================
给定一个包含 n 个整数的排序数组，包含重复，找出给定目标值 target 的起始和结束位置。
如果目标值不在数组中，则返回[-1, -1]

例1:
输入:
[]
9
输出:
[-1,-1]

例2:
输入:
[5, 7, 7, 8, 8, 10]
8
输出:
[3, 4]

挑战
	时间复杂度 O(log n)
*/
/* 
	算法：二分查找模板三
	思路：核心点就是找第一个 target 的索引，和最后一个 target 的索引，
		所以用两次二分搜索分别找第一次和最后一次的位置
*/
func searchRange(A []int, target int) []int {
	n := len(A)
	res := []int{-1,-1}
	if n == 0 {
		return res
	}
	// 先找第一个索引
	L, R := 0, n - 1
	for L + 1 < R {
		mid := L + (R - L) >> 1
		if A[mid] < target {
			L = mid
		} else {
			// 等于时继续往左找就能找到第一个
			R = mid
		}
	}
	// 退出循环会剩下两个元素，搜索最左边的索引
	if A[L] == target {
		res[0] = L
	} else if A[R] == target {
		res[0] = R
	} else {
		// 找不到
		res[0] = -1
		res[1] = -1
		return res
	}

	// 再找最后一个索引
	L, R = 0, n - 1
	for L + 1 < R {
		mid := L + (R - L) >> 1
		if A[mid] <= target {
			// 等于时继续往右找就能找到最后一个
			L = mid
		} else {
			R = mid
		}
	}
	// 退出循环会剩下两个元素，搜索最右边的索引
	if A[R] == target {
		res[1] = R
	} else if A[L] == target {
		res[1] = L
	} else {
		// 找不到
		res[0] = -1
		res[1] = -1
		return res
	}
	return res
}

/* 
===================== 7、搜索插入位置 =====================
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。
如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
你可以假设数组中无重复元素。

示例 1:
输入: [1,3,5,6], 5
输出: 2

示例 2:
输入: [1,3,5,6], 2
输出: 1

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/search-insert-position
*/
/* 
	算法：二分查找模板三
	思路：
		在搜索空间内做比较判断：
			小于等于搜索空间内最左边的返回 L，处于搜索空间内部的返回 R，大于搜索空间最右边的返回 R + 1
*/
func searchInsert(nums []int, target int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	L, R := 0, n - 1
	for L + 1 < R {
		mid := L + (R - L) >> 1
		if nums[mid] < target {
			L = mid
		} else {
			// 等于时往左找
			R = mid
		}
	}
	if nums[L] >= target {
		// 小于等于搜索空间内最左边的
		return L
	} else if nums[R] >= target {
		// 处于搜索空间内部
		return R
	} else {
		// 大于搜索空间最右边的
		return R + 1
	}
}

/* 
===================== 8、搜索二维矩阵 =====================
编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
    每行中的整数从左到右按升序排列。
    每行的第一个整数大于前一行的最后一个整数。

示例 1：
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,50]], target = 3
输出：true

示例 2：
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,50]], target = 13
输出：false

示例 3：
输入：matrix = [], target = 0
输出：false

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/search-a-2d-matrix
*/

/* 
	方法一：二重二分查找
	算法：二分查找模板一 + 二分查找模板三
	思路：
		对行和列进行二分查找
	时间复杂度：O(log mn)
	空间复杂度：O(1)
*/
func searchMatrix(matrix [][]int, target int) bool {
	rn := len(matrix)
	if rn == 0 {
		return false
	}
	cn := len(matrix[0])
	if cn == 0 {
		return false
	}
	// 使用模板一定位行
	rowL, rowR := 0, rn - 1
	for rowL <= rowR {
		rowMid := rowL + ((rowR - rowL) >> 1)
		// 使用模板三定位列
		colL, colR := 0, cn - 1
		for colL + 1 < colR {
			colMid := colL + ((colR - colL) >> 1)
			if matrix[rowMid][colMid] < target {
				colL = colMid
			} else {
				colR = colMid
			}
		}
		if matrix[rowMid][colL] == target || matrix[rowMid][colR] == target {
			return true
		} else {
			// 大于中间行的最右边，则往下半部分找，否则往上半部分找
			if matrix[rowMid][colR] < target {
				rowL = rowMid + 1
			} else {
				rowR = rowMid - 1
			}
		}
	}
	return false
}

/* 
	方法二：二分查找
	算法：二分查找三
	思路：
		把二维数组看做是一维数组，对其使用二分查找。
	时间复杂度：O(log mn)
	空间复杂度：O(1)
*/
func searchMatrix2(matrix [][]int, target int) bool {
	rn := len(matrix)
	if rn == 0 {
		return false
	}
	cn := len(matrix[0])
	if cn == 0 {
		return false
	}
	L, R := 0, rn*cn - 1
	for L + 1 < R {
		mid := L + (R - L) >> 1
		// 获取二维数组对应位置的值
		val := matrix[mid / cn][mid % cn]
		if val < target {
			L = mid
		} else {
			R = mid
		}
	}
	if matrix[L / cn][L % cn] == target ||
		matrix[R / cn][R % cn] == target {
			return true
	} 
	return false
}

/* 
===================== 9、寻找旋转排序数组中的最小值 I =====================
假设按照升序排序的数组在预先未知的某个点上进行了旋转。例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] 。
请找出其中最小的元素。

示例 1：
输入：nums = [3,4,5,1,2]
输出：1

示例 2：
输入：nums = [4,5,6,7,0,1,2]
输出：0

示例 3：
输入：nums = [1]
输出：1

提示：
    1 <= nums.length <= 5000
    -5000 <= nums[i] <= 5000
    nums 中的所有整数都是 唯一 的
    nums 原来是一个升序排序的数组，但在预先未知的某个点上进行了旋转

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array
*/
/* 
	算法：二分查找模板三
	思路：把每一次查找的最后一个值作为 target，当最后一个元素大于等于中间元素时，
		说明最小元素在左边，需要往左移动，否则向右移动，最后比较L、R的值。
*/
func findMin(nums []int) int {
	n := len(nums)
	if n == 1 {
		return nums[0]
	}
	L, R := 0, n - 1
	for L + 1 < R {
		mid := L + (R - L ) >> 1
		// 把每一次查找的最后一个值作为 target，当 target >= nums[mid]时，向左移动
		if nums[mid] <= nums[R] {
			R = mid
		} else {
			L = mid
		}
	}
	if nums[L] > nums[R] {
		return nums[R]
	}
	return nums[L]
}

/* 
===================== 10、寻找旋转排序数组中的最小值 II =====================
假设按照升序排序的数组在预先未知的某个点上进行了旋转。
( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
请找出其中最小的元素。
注意数组中可能存在重复的元素。

示例 1：
输入: [1,3,5]
输出: 1

示例 2：
输入: [2,2,2,0,1]
输出: 0

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii
*/
/* 
	思路：跳过重复元素，把每一次查找的最后一个值作为 target，当最后一个元素大于等于中间元素时，
		说明最小元素在左边，需要往左移动，否则向右移动，最后比较L、R的值。
*/
func findMin2(nums []int) int {
	n := len(nums)
	if n == 1 {
		return nums[0]
	}
	L, R := 0, n - 1
	for L + 1 < R {
		// 跳过重复元素
		for L < R && nums[R] == nums[R - 1] {
			R --
		}
		for L < R && nums[L] == nums[L + 1] {
			L ++
		}
		mid := L + (R - L ) >> 1
		// 把每一次查找的最后一个值作为 target，当 target >= nums[mid]时，向左移动
		if nums[mid] <= nums[R] {
			R = mid
		} else {
			L = mid
		}
	}
	if nums[L] > nums[R] {
		return nums[R]
	}
	return nums[L]
}

/* 
===================== 11、搜索旋转排序数组 II =====================
假设按照升序排序的数组在预先未知的某个点上进行了旋转。
( 例如，数组 [0,0,1,2,2,5,6] 可能变为 [2,5,6,0,0,1,2] )。
编写一个函数来判断给定的目标值是否存在于数组中。若存在返回 true，否则返回 false。

示例 1:
输入: nums = [2,5,6,0,0,1,2], target = 0
输出: true

示例 2:
输入: nums = [2,5,6,0,0,1,2], target = 3
输出: false

进阶:
    这是 搜索旋转排序数组 的延伸题目，本题中的 nums  可能包含重复元素。
    这会影响到程序的时间复杂度吗？会有怎样的影响，为什么？

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii
*/
func search11(nums []int, target int) bool {
	n := len(nums)
	if n == 0 {
		return false
	}
	L, R := 0, n - 1
	for L + 1 < R {
		// 处理重复数字
		for L < R && nums[L] == nums[L + 1] {
			L ++
		}
		for L < R && nums[R] == nums[R - 1] {
			R --
		}
		mid := L + (R - L) >> 1
		 // 相等直接返回
		 if nums[mid] == target {
            return true
        }
		if nums[L] <= nums[mid] {
			if nums[L] <= target && target < nums[mid] {
				R = mid
			} else {
				L = mid
			}
		} else {
			if nums[mid] < target && target <= nums[R] {
				L = mid
			} else {
				R = mid
			}
		}
	}
	if nums[L] == target {
		return true
	}
	if nums[R] == target {
		return true
	}
	return false
}












// ===================== 案列测试 =====================

// 1、二分查找测试
func searchTest() {
	nums := []int{-1,0,3,5,9,12}
	res := search(nums, 9)
	fmt.Println(res)
}

// 2、测试平方根
func mySqrtTest() {
	x := 8
	res := mySqrt2(x)
	fmt.Println(res)
}

// 3、测试猜数字：使用了 leetcode 内部函数，不用测试

// 4、测试搜索旋转排序数组
func search4Test() {
	nums := []int{4,5,6,7,0,1,2}
	res := search4(nums, 0)
	fmt.Println(res)
}

// 5、测试区间搜索
func searchRangeTest() {
	nums := []int{5, 7, 7, 8, 8, 10}
	res := searchRange(nums, 8)
	fmt.Println(res)
}

// 6、测试搜索插入位置
func searchInsertTest() {
	nums := []int{1,3,5,6}
	res := searchInsert(nums, 7)
	fmt.Println(res)
}

// 7、测试搜索二维矩阵
func searchMatrixTest() {
	matrix := [][]int {
		{1,3,5,7},
		{10,11,16,20},
		{23,30,34,50},
	}
	res := searchMatrix2(matrix, 3)
	fmt.Println(res)
}

// 8、测试寻找旋转排序数组中的最小值 I
func findMinTest() {
	nums := []int{1,3,5}
	res := findMin(nums)
	fmt.Println(res)
}

// 9、测试寻找旋转排序数组中的最小值 II
func findMin2Test() {
	nums := []int{1,3,5}
	res := findMin2(nums)
	fmt.Println(res)
}

// 11、测试搜索旋转排序数组 II 
func search11Test() {
	nums := []int{2,5,6,0,0,1,2}
	res := search11(nums, 3)
	fmt.Println(res)
}

func main() {
	// searchTest()
	// mySqrtTest()
	// search4Test()
	// searchRangeTest()
	// searchInsertTest()
	// searchMatrixTest()
	// findMinTest()
	search11Test()
}