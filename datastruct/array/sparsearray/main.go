package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

/*
	稀疏数组
*/

/*
	1、在五子棋程序中，处理存盘退出和续上盘的功能

	0 0 0 0 0 0 0 0 0 0 0
	0 0 0 0 0 0 0 0 0 0 0
	0 0 0 0 0 0 0 0 0 0 0
	0 0 0 0 1 1 1 2 0 0 0
	0 0 0 0 0 2 1 0 0 0 0
	0 0 0 0 2 0 2 0 0 0 0
	0 0 0 0 0 0 0 0 0 0 0
	0 0 0 0 0 0 0 0 0 0 0
	0 0 0 0 0 0 0 0 0 0 0
	0 0 0 0 0 0 0 0 0 0 0
	0 0 0 0 0 0 0 0 0 0 0

	棋盘中 0 表示空白、1表示白子，2表示黑子。正常思维下，如果我们想要存盘，
	则必须把整个 11*11 的二维数组保存起来，但是这么做的话，我们会保存很多
	0，也就是保存了很多无用的空白位置，造成了极大的空间浪费，对此我们可以
	用稀疏数组来优化，稀疏数组的结构如下：

	row  col  val
	11   11   0			<- 首行表示稀疏数组的行数和列数，以及无用元素的值
	3    4    1			<- 从第二行开始，表示 arr[row][col] = val
	3    5    1
	3    6    1
	3    7    2
	4    5    2
	4    6    1
	5    4    2
	5    6    2

	如此，我们只需要用上面的 9*3 稀疏数组就可以表示整个 11*11 棋盘的状态了。
	标准稀疏数组必须存有原始二维数组的 行、列和默认值
*/

// ValNode 棋盘节点
type ValNode struct {
	// 行、列、值
	Row int `json:"row"`
	Col int `json:"col"`
	Val int `json:"val"`
}

// 初始化棋盘
func initChessMap() (chessMap [][]int) {
	// 1、先创建一个初始数组
	chessMap = make([][]int, 11)
	for i := 0; i < len(chessMap); i++ {
		chessMap[i] = make([]int, 11)
	}
	// 2、棋盘赋值
	chessMap[3][4] = 1
	chessMap[3][5] = 1
	chessMap[3][6] = 1
	chessMap[3][7] = 2
	chessMap[4][5] = 2
	chessMap[4][6] = 1
	chessMap[5][4] = 2
	chessMap[5][6] = 2

	// 3、输出棋盘
	fmt.Println("初始棋盘：")
	for _, v := range chessMap {
		for _, v2 := range v {
			fmt.Printf("%d ", v2)
		}
		// 换行输出
		fmt.Println()
	}
	return chessMap
}

// 把棋盘转为稀疏数组
func chessMapToSparseArray(chessMap [][]int) (sparseArr []ValNode) {
	// 4、转成稀疏数组
	// 思路
	// （1）遍历 chessMap，如果发现有一个元素不为 0，则创建一个 node 结构体
	// （2）将 node 结构体放入到对应的切片中
	// 头部添加表示棋盘规模和空白元素的节点
	emptyNode := ValNode{
		Row: len(chessMap),
		Col: len(chessMap[0]),
		Val: 0,
	}
	sparseArr = append(sparseArr, emptyNode)
	// 扫描棋盘，把有效元素添加进稀疏数组中
	for i, v := range chessMap {
		for j, v2 := range v {
			if v2 != 0 {
				valNode := ValNode{
					Row: i,
					Col: j,
					Val: v2,
				}
				sparseArr = append(sparseArr, valNode)
			}
		}
	}
	// 遍历稀疏数组
	fmt.Println("棋盘转稀疏数组：")
	for _, v := range sparseArr {
		fmt.Println(v)
	}
	return
}

// 把存有棋盘信息的稀疏数组存盘
func saveSparseArrayToDisk(filePath string, valNodes []ValNode) (err error) {
	// 1、打开文件，如果文件不存在则创建
	file, err := os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0755)
	defer file.Close()
	if err != nil {
		fmt.Println("os.OpenFile err:", err)
		return
	}

	// 2、创建 Writer
	writer := bufio.NewWriter(file)

	// 3、把稀疏数组转为 string
	str := ""
	for _, v := range valNodes {
		str += fmt.Sprintf("%d %d %d\n", v.Row, v.Col, v.Val)
	}
	// 4、把稀疏数组内容写入文件
	_, err = writer.Write([]byte(str))
	if err != nil {
		fmt.Println("writer.Write err:", err)
		return
	}
	// 5、刷新缓冲区
	err = writer.Flush()
	if err != nil {
		fmt.Println("writer.Flush err:", err)
		return
	}
	fmt.Println("稀疏数组存盘：保存成功")
	return
}

// 从磁盘中读取存有棋盘信息的稀疏数组
func readSparseArrayFromDisk(filePath string) (sparseArr []ValNode, err error) {
	// 1、打开文件
	file, err := os.OpenFile(filePath, os.O_RDONLY, 0755)
	if err != nil {
		fmt.Println("os.OpenFile err:", err)
		return
	}
	// 2、创建 Reader
	reader := bufio.NewReader(file)

	// 3、读取文件内容
	for {
		// 4、从文件中读取内容
		data, err := reader.ReadBytes('\n')
		if err != nil {
			// 读取到文件末尾，则跳出循环
			if err != io.EOF {
				fmt.Println("reader.ReadString('\n') err:", err)
			}
			break
		}
		// 4、解析字符串，转为 ValNode
		str := string(data)
		// 5、去除 \n
		str = strings.Trim(str, "\n")
		// 去除最后一行的影响
		if len(str) == 0 {
			break
		}
		// 6、字符串转为数值数组
		arr := strings.Split(str, " ")
		row, err := strconv.Atoi(arr[0])
		if err != nil {
			fmt.Println("strconv.Atoi(arr[0]) err:", err)
			return nil, err
		}
		col, err := strconv.Atoi(arr[1])
		if err != nil {
			fmt.Println("strconv.Atoi(arr[1]) err:", err)
			return nil, err
		}
		val, err := strconv.Atoi(arr[2])
		if err != nil {
			fmt.Println("strconv.Atoi(arr[2]) err:", err)
			return nil, err
		}
		// 7、构建 ValNode
		node := ValNode{
			Row: row,
			Col: col,
			Val: val,
		}
		// 8、组成稀疏数组
		sparseArr = append(sparseArr, node)
	}

	fmt.Println("稀疏数组读取：")
	for _, v := range sparseArr {
		fmt.Println(v)
	}
	return
}

// 把稀疏数组还原为棋盘
func sparseArrayToChessMap(sparseArr []ValNode) (chessMap [][]int, err error) {
	// 5、把稀疏数组转为二维棋盘
	if len(sparseArr) == 0 {
		fmt.Println("读取不到棋盘数据")
		return
	}
	// 取出第一个元素，该元素记录有棋盘的行、列和默认值信息
	node := sparseArr[0]
	// 按行、列和默认值创建棋盘
	chessMap = make([][]int, node.Row)
	for i := 0; i < node.Row; i++ {
		chessMap[i] = make([]int, node.Col)
		for j := 0; j < node.Col; j++ {
			// 初始值赋值
			chessMap[i][j] = node.Val
		}
	}

	// 把稀疏数组中的值复原到棋盘中
	for i := 1; i < len(sparseArr); i++ {
		n := sparseArr[i]
		chessMap[n.Row][n.Col] = n.Val
	}

	// 输出复原后的棋盘
	fmt.Println("棋盘复原：")
	for _, v := range chessMap {
		for _, v2 := range v {
			fmt.Printf("%d ", v2)
		}
		// 换行输出
		fmt.Println()
	}
	return
}

func main() {
	// 1、初始化棋盘数据
	chessMap := initChessMap()
	// 2、把棋盘数据转为稀疏数组
	sparseArr := chessMapToSparseArray(chessMap)
	// 3、把带有棋盘数据的稀疏数组存入磁盘文件
	filePath := "D:/golang_test/chessMap.data"
	err := saveSparseArrayToDisk(filePath, sparseArr)
	if err != nil {
		fmt.Println("saveSparseArrayToDisk err:", err)
		return
	}
	// 4、从磁盘文件中读取带有棋盘数据的稀疏数组
	arr, err := readSparseArrayFromDisk(filePath)
	if err != nil {
		fmt.Println("readSparseArrayFromDisk err:", err)
		return
	}
	// 5、把稀疏数组复原为棋盘数据
	_, err = sparseArrayToChessMap(arr)
	if err != nil {
		fmt.Println("sparseArrayToChessMap err:", err)
		return
	}
}
