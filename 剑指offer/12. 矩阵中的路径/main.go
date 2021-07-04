package main

import "fmt"

/*
============== 剑指 Offer 12. 矩阵中的路径 ==============
给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于
网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水
平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

例如，在下面的 3×4 的矩阵中包含单词 "ABCCED"（单词中的字母已标出）。

示例 1：

输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]],
	word = "ABCCED"
输出：true
说明：
	A--B--C  E
	   	  |
	S  F  C  S
		  |
	A  D--E  E

示例 2：
输入：board = [["a","b"],["c","d"]], word = "abcd"
输出：false

提示：
    1 <= board.length <= 200
    1 <= board[i].length <= 200
    board 和 word 仅由大小写英文字母组成

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof
*/
/*
	方法一：回溯法
	思路：
		首先，在网格中任选一个格子作为起点，假设矩阵中某个格子 board[r][c]
		的字符为 ch，并且这个格子将对于与路径上的第 i 个字符。如果路径上的第
		i 个不是 ch，那么 board[r][c] 不可能在路径上的第 j 个位置。否则到
		board[r][c] 的相邻格子去寻找路径上的第 i+1 个字符。

		除了矩阵边界上的格子之外，其他格子都有 4 个相邻的格子，重复上述过程，
		知道路径上的所有字符都在矩阵中找到相应的位置。

		由于回溯法的特性，路径可以被看做一个栈。当矩阵中定位了路径中前 n 个字符
		的位置之后，在与第 n 个字符对应的格子的周围都没有找到第 i+1 个字符时，
		这时候只好在路径上回到第 n-1 个字符，重新定位第 n 个字符。

		注意：由于路径不能重复进入矩阵的同一个格子，所以我们记录当前路径在矩阵
		中所走过的痕迹，当路径回溯时，痕迹也要移除。
	时间复杂度：O((3^)*mn)
		m、n 分别是矩阵的行数和列数，k 是匹配字符串的长度。
		最差情况下，需要遍历矩阵中长度为 K 字符串的所有方案，时间复杂度
		为 O(3^k)；矩阵中共有 mn 个起点，时间复杂度为 O(mn)。
	空间复杂度：O(m*n)
		我们需要用一个 m*n 的二维数组来记录路径的访问痕迹。
*/
func exist(board [][]byte, word string) bool {
	m := len(board)
	if m == 0 {
		return false
	}
	n := len(board[0])
	if n == 0 {
		return false
	}
	visited := make([][]bool, m)
	for i := 0; i < m; i++ {
		visited[i] = make([]bool, n)
	}
	index := 0
	// 任选一个格子作为路径的起点
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			// 只要完全匹配上一次就返回
			if hasCode(board, i, j, visited, word, &index) {
				return true
			}
		}
	}
	return false
}

// 对路径上的第 index 个字符进行匹配
// board 矩阵
// i、j 当前在矩阵中的下标
// visited 当前路径在矩阵中走过的痕迹
// word 目标路径
// *index 当前要匹配的路径字符的下标
func hasCode(board [][]byte, i, j int, visited [][]bool,
	word string, index *int) bool {
	// 字符串路径的所有字符都已匹配已完成
	if (*index) == len(word) {
		return true
	}

	has := false
	if i >= 0 && i < len(board) && j >= 0 && j < len(board[0]) &&
		!visited[i][j] && word[(*index)] == board[i][j] {
		// 标记当前位置为已访问
		visited[i][j] = true
		// 下一个字符的下标
		(*index)++

		// 尝试在当前位置的邻居节点中去匹配路径的下一个字符
		up := hasCode(board, i-1, j, visited, word, index)
		down := hasCode(board, i+1, j, visited, word, index)
		left := hasCode(board, i, j-1, visited, word, index)
		right := hasCode(board, i, j+1, visited, word, index)

		has = up || down || left || right
		// 匹配失败，进行回溯
		if !has {
			(*index)--
			visited[i][j] = false
		}
	}
	return has
}

/*
	方法二： 深度优先搜索（DFS）+ 剪枝
	思路：
		深度优先搜索：
			可以理解为暴力法遍历矩阵中所有字符串可能性。DFS 通过递归，先朝一
			个方向搜到底，再回溯至上个节点，沿另一个方向搜索，以此类推。
		剪枝：
			在搜索中，遇到 这条路不可能和目标字符串匹配成功 的情况（例如：
			此矩阵元素和目标字符不同、此元素已被访问），则应立即返回，称之
			为 可行性剪枝 。
		DFS 解析：
			递归参数：
				当前元素在矩阵 board 中的行列索引 i 和 j ，当前目标字符在
				word 中的索引 k 。
			终止条件：
				返回 false：
					(1) 行或列索引越界 或 (2) 当前矩阵元素与目标字符不同
					或 (3) 当前矩阵元素已访问过 （ (3) 可合并至 (2) ） 。
				返回 true ：
					k = len(word) - 1 ，即字符串 word 已全部匹配。
			递推工作：
				标记当前矩阵元素：
					将 board[i][j] 修改为 空字符 '' ，代表此元素已访问过，
					防止之后搜索时重复访问。
				搜索下一单元格：
					朝当前元素的 上、下、左、右 四个方向开启下层递归，使用
					或 连接 （代表只需找到一条可行路径就直接返回，不再做后
					续 DFS ），并记录结果至 res 。
				还原当前矩阵元素：
					将 board[i][j] 元素还原至初始值，即 word[k] 。
			返回值：
				返回布尔量 res ，代表是否搜索到目标字符串。
	复杂度分析：
    	M,N 分别为矩阵行列大小， K 为字符串 word 长度。
    时间复杂度 O((3^k)MN) ：
		最差情况下，需要遍历矩阵中长度为 K 字符串的所有方案，时间复杂度为
		O(3K)；矩阵中共有 MN 个起点，时间复杂度为 O(MN) 。
        方案数计算：
			设字符串长度为 K ，搜索中每个字符有上、下、左、右四个方向可以选
			择，舍弃回头（上个字符）的方向，剩下 3 种选择，因此方案数的复杂
			度为 O(3^K) 。
    空间复杂度 O(K)：
		搜索过程中的递归深度不超过 K ，因此系统因函数调用累计使用的栈空间
		占用 O(K) （因为函数返回后，系统调用的栈空间会释放）。最坏情况
		下 K=MN ，递归深度为 MN ，此时系统栈使用 O(MN) 的额外空间。
*/
func exist2(board [][]byte, word string) bool {
	m := len(board)
	if m == 0 {
		return false
	}
	n := len(board[0])
	if n == 0 {
		return false
	}
	visited := make([][]bool, m)
	for i := 0; i < m; i++ {
		visited[i] = make([]bool, n)
	}
	k := 0
	// 任选一个格子作为路径的起点
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			// 只要完全匹配上一次就返回
			if DFS(board, word, i, j, k) {
				return true
			}
		}
	}
	return false
}
func DFS(board [][]byte, word string, i, j, k int) bool {
	// 这里不用判断是否访问，如果已访问的话，board[i][j] = '0'，必不
	// 可能与 word[k] 相等
	if i < 0 || i >= len(board) || j < 0 || j >= len(board[0]) ||
		board[i][j] != word[k] {
		return false
	}
	// 所有字符已匹配完成
	if k == len(word) - 1 {
		return true
	}
	// 标记为已访问
	board[i][j] = '0'
	ans := DFS(board, word, i - 1, j, k + 1) ||
			DFS(board, word, i + 1, j, k + 1) ||
			DFS(board, word, i, j - 1, k + 1) ||
			DFS(board, word, i, j + 1, k + 1)
	// 回溯，把 board[i][j] 设置为原来的值
	board[i][j] = word[k]
	return ans
}

func main() {
	// board := [][]byte{
	// 	{'a', 'b', 't', 'g'},
	// 	{'c', 'f', 'c', 's'},
	// 	{'j', 'd', 'e', 'h'},
	// }
	board := [][]byte {{'a', 'a'}}
	word := "aaa"
	// res := exist(board, word)
	res := exist2(board, word)
	fmt.Println(res)
}
