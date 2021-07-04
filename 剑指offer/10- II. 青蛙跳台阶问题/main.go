package main

/*
============== 剑指 Offer 10- II. 青蛙跳台阶问题 ==============
一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶
总共有多少种跳法。
答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

示例 1：
输入：n = 2
输出：2

示例 2：
输入：n = 7
输出：21

示例 3：
输入：n = 0
输出：1

提示：
    0 <= n <= 100

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof
*/

/*
	方法一：递归
	思路：
		我们先分析问题：
			我们设 f(n) 为青蛙跳上 n 阶台阶的跳法数量，则有
				n = 0，f(n) = 1
					只有一种跳法，那就是不跳。
				n = 1，f(n) = 1
					只有一种跳法，只跳一级。
				n = 2, f(n) = 2
					一种是分两次跳，每次跳 1 级；另一种是一次跳 2 级。
				当 n > 2 时：
					第一次跳的时候就有两种不同的选择：
						1、第一次只跳 1 级，此时跳法数目等于后面剩下 n-1 级
							台阶的跳法数目，即为 f(n-1);
						2、第一次跳 2 级，此时跳法数目等于后面剩下 n-2 级
							台阶的跳法数目，即为 f(n-2)。
					所以 n > 2 时，f(n) = f(n-1) + f(n-2)。
		由此分析可知，本题与斐波那契数列题目一样。
	时间复杂度：O(2^n)
	空间复杂度：O(n)
*/
func numWays(n int) int {
	if n == 0 {
		return 1
	}
	if n == 1 {
		return 1
	}
	return (numWays(n-1) + numWays(n-2)) % 1000000007
}

/* 
	方法二：循环
	思路：
		使用循环“自底向上”的方式实现递归计算的逆过程，计算过程中记录中间结果。
	时间复杂度：O(n)
	空间复杂度：O(n)
*/
func numWays2(n int) int {
	if n < 2 {
		return 1
	}
	ans := make([]int, n + 1)
	ans[0], ans[1] = 1, 1
	for i := 2; i <= n; i ++ {
		ans[i] = (ans[i-1] + ans[i-2]) % 1000000007
	}
	return ans[n]
}

/* 
	方法三：动态规划【空间优化】
	思路：
		状态定义：设 dp 为一维数组，其中 dp[i] 的值代表青蛙跳第 i 个台阶的跳
			法数量。
		起始条件：
			dp[0] = 1, dp[1] = 1
		状态转移方程：
			dp[i] = dp[i-1] + dp[i-2]，即 f(n) = f(n-1) + f(n-2)
		终止条件：
			dp(n)
	空间优化：
		在方法二的循环处理过程中，我们需要记录每一步运算的结果来避免重复运算，
		但通过观察状态转移方程 dp[i] = dp[i-1] + dp[i-2] 发现，
		对于每一个状态 dp[i] 来说，能够影响其状态变化的只有 dp[i-1] 和
		dp[i-2]，我们无需额外记录 dp[0] ~ dp[i-3] 的中间状态值。
		如此我们只需用两个变量来记录 dp[i-1] 和 dp[i-2] 就能完成所有斐波那
		契数的状态转移计算。
	循环求余法：
		大数越界：
			随着 n 增大, f(n) 会超过 Int32 甚至 Int64 的取值范围，导致最终
			的返回值错误。
		求余运算规则：
			设正整数 x、y、p，求余符号为 ⊙，则有 (x+y)⊙p=(x⊙p+y⊙p)⊙p
		解析：
			根据以上规则，可推出 f(n)⊙p=[f(n−1)⊙p+f(n−2)⊙p]⊙p ，
			从而可以在循环过程中每次计算 sum=(a+b)⊙1000000007，此操作
			与最终返回前取余等价。
	时间复杂度：O(n)
	空间复杂度：O(1)
*/
func numWays3(n int) int {
	if n == 0 {
		return 1
	}
	if n == 1 {
		return 1
	}
	a, b := 1, 1
	ans := 0
	for i := 2; i <= n; i++ {
		ans = (a + b) % 1000000007
		a = b
		b = ans
	}
	return ans
}