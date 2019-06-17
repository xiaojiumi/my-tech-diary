#[动态规划](https://github.com/CyC2018/CS-Notes/blob/master/notes/Leetcode%20%E9%A2%98%E8%A7%A3%20-%20%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92.md)

[斐波那契数列](https://github.com/CyC2018/CS-Notes/blob/master/notes/Leetcode%20%E9%A2%98%E8%A7%A3%20-%20%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92.md#%E6%96%90%E6%B3%A2%E9%82%A3%E5%A5%91%E6%95%B0%E5%88%97) 

#[70. Climbing Stairs (Easy)](https://leetcode.com/problems/climbing-stairs/description/)

题目描述：有 N 阶楼梯，每次可以上一阶或者两阶，求有多少种上楼梯的方法。

定义一个数组 dp 存储上楼梯的方法数（为了方便讨论，数组下标从 1 开始），dp[i] 表示走到第 i 个楼梯的方法数目。

第 i 个楼梯可以从第 i-1 和 i-2 个楼梯再走一步到达，走到第 i 个楼梯的方法数为走到第 i-1 和第 i-2 个楼梯的方法数之和。

[![img](https://github.com/CyC2018/CS-Notes/raw/master/notes/pics/14fe1e71-8518-458f-a220-116003061a83.png)](https://github.com/CyC2018/CS-Notes/blob/master/notes/pics/14fe1e71-8518-458f-a220-116003061a83.png)

 

考虑到 dp[i] 只与 dp[i - 1] 和 dp[i - 2] 有关，因此可以只用两个变量来存储 dp[i - 1] 和 dp[i - 2]，使得原来的 O(N) 空间复杂度优化为 O(1) 复杂度。

```
public int climbStairs(int n) {
    if (n <= 2) {
        return n;
    }
    int pre2 = 1, pre1 = 2;
    for (int i = 2; i < n; i++) {
        int cur = pre1 + pre2;
        pre2 = pre1;
        pre1 = cur;
    }
    return pre1;
}
```



#[213. House Robber II (Medium)](https://leetcode.com/problems/house-robber-ii/description/)

```
   /**
     * 解题思路：
     * 难点->最后一个既然是和第一个相连的，不然一个动态规划等式就能解决了
     * dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
     * 突破这个难点，使用两个规划数组，一个从0开始，n-1结束，另一个从1开始，n结束
     * 求出两个数组的最大值
     *
     * @param nums
     * @return
     */
public int rob(int[] nums) {
    if (nums == null || nums.length == 0) {
        return 0;
    }
    int n = nums.length;
    if (n == 1) {
        return nums[0];
    }
    return Math.max(rob(nums, 0, n - 2), rob(nums, 1, n - 1));
}

private int rob(int[] nums, int first, int last) {
    int pre2 = 0, pre1 = 0;
    for (int i = first; i <= last; i++) {
        int cur = Math.max(pre1, pre2 + nums[i]);
        pre2 = pre1;
        pre1 = cur;
    }
    return pre1;
}
```


#[64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)



给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

示例:

输入:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 7
解释: 因为路径 1→3→1→1→1 的总和最小。

方法 2：二维动态规划
算法

我们新建一个额外的 dpdp 数组，与原矩阵大小相同。在这个矩阵中，dp(i, j)dp(i,j) 表示从坐标 (i, j)(i,j) 到右下角的最小路径权值。我们初始化右下角的 dpdp 值为对应的原矩阵值，然后去填整个矩阵，对于每个元素考虑移动到右边或者下面，因此获得最小路径和我们有如下递推公式：

![1559972624277](C:\Users\Song\AppData\Local\Temp\1559972624277.png)

```java
public class Solution {
    public int minPathSum(int[][] grid) {
        int[][] dp = new int[grid.length][grid[0].length];
        for (int i = grid.length - 1; i >= 0; i--) {
            for (int j = grid[0].length - 1; j >= 0; j--) {
                if(i == grid.length - 1 && j != grid[0].length - 1)
                    dp[i][j] = grid[i][j] +  dp[i][j + 1];
                else if(j == grid[0].length - 1 && i != grid.length - 1)
                    dp[i][j] = grid[i][j] + dp[i + 1][j];
                else if(j != grid[0].length - 1 && i != grid.length - 1)
                    dp[i][j] = grid[i][j] + Math.min(dp[i + 1][j], dp[i][j + 1]);
                else
                    dp[i][j] = grid[i][j];
            }
        }
        return dp[0][0];
    }
}


```

方法 3：一维动态规划
算法

在上个解法中，我们可以用一个一维数组来代替二维数组，dpdp 数组的大小和行大小相同。这是因为对于某个固定状态，只需要考虑下方和右侧的节点。首先初始化 dpdp 数组最后一个元素是右下角的元素值，然后我们向左移更新每个 dp(j)dp(j) 为：

![1559972730254](C:\Users\Song\AppData\Local\Temp\1559972730254.png)

我们对于每一行都重复这个过程，然后向上一行移动，计算完成后 dp(0)dp(0) 就是最后的结果。

```java
public class Solution {
   public int minPathSum(int[][] grid) {
       int[] dp = new int[grid[0].length];
       for (int i = grid.length - 1; i >= 0; i--) {
           for (int j = grid[0].length - 1; j >= 0; j--) {
               if(i == grid.length - 1 && j != grid[0].length - 1)
                   dp[j] = grid[i][j] +  dp[j + 1];
               else if(j == grid[0].length - 1 && i != grid.length - 1)
                   dp[j] = grid[i][j] + dp[j];
               else if(j != grid[0].length - 1 && i != grid.length - 1)
                   dp[j] = grid[i][j] + Math.min(dp[j], dp[j + 1]);
               else
                   dp[j] = grid[i][j];
           }
       }
       return dp[0];
   }
}
```



#[62. Unique Paths (Medium)](https://leetcode.com/problems/unique-paths/description/)

题目描述：统计从矩阵左上角到右下角的路径总数，每次只能向右或者向下移动。





[![img](https://github.com/CyC2018/CS-Notes/raw/master/notes/pics/dc82f0f3-c1d4-4ac8-90ac-d5b32a9bd75a.jpg)](https://github.com/CyC2018/CS-Notes/blob/master/notes/pics/dc82f0f3-c1d4-4ac8-90ac-d5b32a9bd75a.jpg)

 

[![img](https://github.com/CyC2018/CS-Notes/raw/master/notes/pics/dc82f0f3-c1d4-4ac8-90ac-d5b32a9bd75a.jpg)](https://github.com/CyC2018/CS-Notes/blob/master/notes/pics/dc82f0f3-c1d4-4ac8-90ac-d5b32a9bd75a.jpg)

 递推式已经出来了，就是res[i][j]=res[i-1][j]+res[i][j-1]，这样我们就可以用一个数组来保存历史信息，也就是在i行j列的路径数，这样每次就不需要重复计算，从而降低复杂度。用动态规划我们只需要对所有格子进行扫描一次，到了最后一个得到的结果就是总的路径数，所以时间复杂度是O(m*n)。而对于空间可以看出我们每次只需要用到上一行当前列，以及前一列当前行的信息，我们只需要用一个一维数组存上一行的信息即可，然后扫过来依次更替掉上一行对应列的信息即可（因为所需要用到的信息都还没被更替掉），所以空间复杂度是O(n)（其实如果要更加严谨，我们可以去行和列中小的那个，然后把小的放在内层循环，这样空间复杂度就是O(min(m,n))

```
public int uniquePaths(int m, int n) {
    if(m<=0 || n<=0)
        return 0;
    int[] res = new int[n];     //如果不好理解就用二维数组，这样空间复杂度会增加
    res[0] = 1;
    for(int i=0;i<m;i++)
    {
        for(int j=1;j<n;j++)
        {
           res[j] += res[j-1];
        }
    }
    return res[n-1];
}
```

也可以直接用数学公式求解，这是一个组合问题。机器人总共移动的次数 S=m+n-2，向下移动的次数 D=m-1，那么问题可以看成从 S 中取出 D 个位置的组合数量，这个问题的解为 C(S, D)。

```
public int uniquePaths(int m, int n) 
    {
        int num = Math.min(m, n)-1;
        return (int)(factorial(m+n-2,num)/factorial(num, num));
    }
    
    public long factorial(int start, int count) //这里必须用long，int很容易越界
    {
        long num = 1;
        while (count-->0) num *= start--;
        return num;        
    }
```



#[303. Range Sum Query - Immutable (Easy)](https://leetcode.com/problems/range-sum-query-immutable/description/)

```
Given nums = [-2, 0, 3, -5, 2, -1]

sumRange(0, 2) -> 1
sumRange(2, 5) -> -1
sumRange(0, 5) -> -3
```

求区间 i ~ j 的和，可以转换为 sum[j + 1] - sum[i]，其中 sum[i] 为 0 ~ i - 1 的和。

```
class NumArray {

    private int[] sums;

    public NumArray(int[] nums) {
        sums = new int[nums.length + 1];
        for (int i = 1; i <= nums.length; i++) {
            sums[i] = sums[i - 1] + nums[i - 1];
        }
    }

    public int sumRange(int i, int j) {
        return sums[j + 1] - sums[i];
    }
}
```



#[413. Arithmetic Slices (Medium)](https://leetcode.com/problems/arithmetic-slices/description/)

```
A = [0, 1, 2, 3, 4]

return: 6, for 3 arithmetic slices in A:

[0, 1, 2],
[1, 2, 3],
[0, 1, 2, 3],
[0, 1, 2, 3, 4],
[ 1, 2, 3, 4],
[2, 3, 4]
```

dp[i] 表示以 A[i] 为结尾的等差递增子区间的个数。

当 A[i] - A[i-1] == A[i-1] - A[i-2]，那么 [A[i-2], A[i-1], A[i]] 构成一个等差递增子区间。而且在以 A[i-1] 为结尾的递增子区间的后面再加上一个 A[i]，一样可以构成新的递增子区间。

```
dp[2] = 1
    [0, 1, 2]
dp[3] = dp[2] + 1 = 2
    [0, 1, 2, 3], // [0, 1, 2] 之后加一个 3
    [1, 2, 3]     // 新的递增子区间
dp[4] = dp[3] + 1 = 3
    [0, 1, 2, 3, 4], // [0, 1, 2, 3] 之后加一个 4
    [1, 2, 3, 4],    // [1, 2, 3] 之后加一个 4
    [2, 3, 4]        // 新的递增子区间
```

综上，在 A[i] - A[i-1] == A[i-1] - A[i-2] 时，dp[i] = dp[i-1] + 1。

因为递增子区间不一定以最后一个元素为结尾，可以是任意一个元素结尾，因此需要返回 dp 数组累加的结果。

```
public int numberOfArithmeticSlices(int[] A) {
    if (A == null || A.length == 0) {
        return 0;
    }
    int n = A.length;
    int[] dp = new int[n];
    for (int i = 2; i < n; i++) {
        if (A[i] - A[i - 1] == A[i - 1] - A[i - 2]) {
            dp[i] = dp[i - 1] + 1;
        }
    }
    int total = 0;
    for (int cnt : dp) {
        total += cnt;
    }
    return total;
}
```



#[343. Integer Break (Medim)](https://leetcode.com/problems/integer-break/description/)

题目描述：For example, given n = 2, return 1 (2 = 1 + 1); given n = 10, return 36 (10 = 3 + 3 + 4).

```
public int integerBreak(int n) {
    int[] dp = new int[n + 1];
    dp[1] = 1;
    for (int i = 2; i <= n; i++) {
        for (int j = 1; j <= i - 1; j++) {
            dp[i] = Math.max(dp[i], Math.max(j * dp[i - j], j * (i - j)));
        }
    }
    return dp[n];
}
```



# [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。

示例 1:

输入: n = 12
输出: 3 
解释: 12 = 4 + 4 + 4.
示例 2:

输入: n = 13
输出: 2
解释: 13 = 4 + 9.

可以将每个整数看成图中的一个节点，如果两个整数之差为一个平方数，那么这两个整数所在的节点就有一条边。

要求解最小的平方数数量，就是求解从节点 n 到节点 0 的最短路径。

本题也可以用动态规划求解，在之后动态规划部分中会再次出现。

```
public int numSquares(int n) {
    List<Integer> squares = generateSquares(n);
    Queue<Integer> queue = new LinkedList<>();
    boolean[] marked = new boolean[n + 1];
    queue.add(n);
    marked[n] = true;
    int level = 0;
    while (!queue.isEmpty()) {
        int size = queue.size();
        level++;
        while (size-- > 0) {
            int cur = queue.poll();
            for (int s : squares) {
                int next = cur - s;
                if (next < 0) {
                    break;
                }
                if (next == 0) {
                    return level;
                }
                if (marked[next]) {
                    continue;
                }
                marked[next] = true;
                queue.add(next);
            }
        }
    }
    return n;
}

/**
 * 生成小于 n 的平方数序列
 * @return 1,4,9,...
 */
private List<Integer> generateSquares(int n) {
    List<Integer> squares = new ArrayList<>();
    int square = 1;
    int diff = 3;
    while (square <= n) {
        squares.add(square);
        square += diff;
        diff += 2;
    }
    return squares;
}
```

动态规划

```
public int numSquares(int n) {
    List<Integer> squareList = generateSquareList(n);
    int[] dp = new int[n + 1];
    for (int i = 1; i <= n; i++) {
        int min = Integer.MAX_VALUE;
        for (int square : squareList) {
            if (square > i) {
                break;
            }
            min = Math.min(min, dp[i - square] + 1);
        }
        dp[i] = min;
    }
    return dp[n];
}

private List<Integer> generateSquareList(int n) {
    List<Integer> squareList = new ArrayList<>();
    int diff = 3;
    int square = 1;
    while (square <= n) {
        squareList.add(square);
        square += diff;
        diff += 2;
    }
    return squareList;
}
```



#[91. Decode Ways (Medium)](https://leetcode.com/problems/decode-ways/description/)

题目描述：Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12).

思路：

可以看作一个动态规划的问题，

首先，如果第一位上是0，那么无法转码，返回0；

其次，每次转码，都可以看看第i位跟第i-1位的数字是否小于等于26，如果符合的话，我们在i-2位上解码；如果第i位不等于0的话，说明我们相当于在第i-1位上解码。如果两个都符合，相当于：dp[i] = dp[i-1] + dp[i-2];

```
public int numDecodings(String s) {
    if (s == null || s.length() == 0) {
        return 0;
    }
    int n = s.length();
    int[] dp = new int[n + 1];
    dp[0] = 1;
    dp[1] = s.charAt(0) == '0' ? 0 : 1;
    for (int i = 2; i <= n; i++) {
        int one = Integer.valueOf(s.substring(i - 1, i));
        if (one != 0) {
            dp[i] += dp[i - 1];
        }
        if (s.charAt(i - 2) == '0') {
            continue;
        }
        int two = Integer.valueOf(s.substring(i - 2, i));
        if (two <= 26) {
            dp[i] += dp[i - 2];
        }
    }
    return dp[n];
}
```


# [300. 最长上升子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

给定一个无序的整数数组，找到其中最长上升子序列的长度。

示例:

输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
说明:

可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。
你算法的时间复杂度应该为 O(n2) 。
进阶: 你能将算法的时间复杂度降低到 O(n log n) 吗?



典型的动态规划题目，定义一个数组dp，其中dp[i]代表以第num[i]为结尾取得的最长长度，最后返回最大的那个dp[i]就行了，

所以这道题的重点在于怎么求得dp[i]。一开始我们先把dp全部元素都初始化为1，因为dp[i]无论如何最少的长度都是1（只有本身一个元素的序列），然后对于每个num[i]我们可以通过遍历num[0]~num[i-1]，如果在这个区间找到一个num[j]比num[i]小，那么开始比较dp[j]+1和dp[i]的大小，如果dp[j]+1>dp[i]则更新dp[i]，所以转移方程为dp[i] = max(dp[j]+1,dp[i])。



```java
 public int lengthOfLIS(int[] nums) {
        if(nums==null||nums.length==0)
            return 0;
        
        int n=nums.length;
        int[] dp=new int[n];
        int res=1;
        for(int i=0;i<n;i++){
             int max=1;
            for(int j=0;j<i;j++){
                if(nums[i]>nums[j]){
                    max=Math.max(max,dp[j]+1);
                }
                if(max>res)
                    res=max;
            }
            dp[i]=max;
        }
           
           return res;     
    }
```



#[376. Wiggle Subsequence (Medium)](https://leetcode.com/problems/wiggle-subsequence/description/)

```
Input: [1,7,4,9,2,5]
Output: 6
The entire sequence is a wiggle sequence.

Input: [1,17,5,10,13,15,10,5,16,8]
Output: 7
There are several subsequences that achieve this length. One is [1,17,10,13,10,16,8].

Input: [1,2,3,4,5,6,7,8,9]
Output: 2
```

要求：使用 O(N) 时间复杂度求解。

```
public int wiggleMaxLength(int[] nums) {
    if (nums == null || nums.length == 0) {
        return 0;
    }
    int up = 1, down = 1;
    for (int i = 1; i < nums.length; i++) {
        if (nums[i] > nums[i - 1]) {
            up = down + 1;
        } else if (nums[i] < nums[i - 1]) {
            down = up + 1;
        }
    }
    return Math.max(up, down);
}
```



#最长公共子序列

对于两个子序列 S1 和 S2，找出它们最长的公共子序列。

定义一个二维数组 dp 用来存储最长公共子序列的长度，其中 dp[i][j] 表示 S1 的前 i 个字符与 S2 的前 j 个字符最长公共子序列的长度。考虑 S1i 与 S2j 值是否相等，分为两种情况：

- 当 S1i==S2j 时，那么就能在 S1 的前 i-1 个字符与 S2 的前 j-1 个字符最长公共子序列的基础上再加上 S1i 这个值，最长公共子序列长度加 1，即 dp[i][j] = dp[i-1][j-1] + 1。
- 当 S1i != S2j 时，此时最长公共子序列为 S1 的前 i-1 个字符和 S2 的前 j 个字符最长公共子序列，或者 S1 的前 i 个字符和 S2 的前 j-1 个字符最长公共子序列，取它们的最大者，即 dp[i][j] = max{ dp[i-1][j], dp[i][j-1] }。

综上，最长公共子序列的状态转移方程为：

[![img](https://github.com/CyC2018/CS-Notes/raw/master/notes/pics/ecd89a22-c075-4716-8423-e0ba89230e9a.jpg)](https://github.com/CyC2018/CS-Notes/blob/master/notes/pics/ecd89a22-c075-4716-8423-e0ba89230e9a.jpg)

 

对于长度为 N 的序列 S1 和长度为 M 的序列 S2，dp[N][M] 就是序列 S1 和序列 S2 的最长公共子序列长度。

与最长递增子序列相比，最长公共子序列有以下不同点：

- 针对的是两个序列，求它们的最长公共子序列。
- 在最长递增子序列中，dp[i] 表示以 Si 为结尾的最长递增子序列长度，子序列必须包含 Si ；在最长公共子序列中，dp[i][j] 表示 S1 中前 i 个字符与 S2 中前 j 个字符的最长公共子序列长度，不一定包含 S1i 和 S2j。
- 在求最终解时，最长公共子序列中 dp[N][M] 就是最终解，而最长递增子序列中 dp[N] 不是最终解，因为以 SN 为结尾的最长递增子序列不一定是整个序列最长递增子序列，需要遍历一遍 dp 数组找到最大者。

```
public int lengthOfLCS(int[] nums1, int[] nums2) {
    int n1 = nums1.length, n2 = nums2.length;
    int[][] dp = new int[n1 + 1][n2 + 1];
    for (int i = 1; i <= n1; i++) {
        for (int j = 1; j <= n2; j++) {
            if (nums1[i - 1] == nums2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[n1][n2];
}
```

#[416. Partition Equal Subset Sum (Medium)](https://leetcode.com/problems/partition-equal-subset-sum/description/)

```
Input: [1, 5, 11, 5]

Output: true

Explanation: The array can be partitioned as [1, 5, 5] and [11].
```

可以看成一个背包大小为 sum/2 的 0-1 背包问题。



```
/**
大佬给出的一维数组动态规划有点懵逼。
这里先给出二维数组的动态规划，然后给出转化为一维数组的方法。理解起来相信非常容易。
所以这里会给出三个版本的代码：
二维数组动态规划
一维数组动态规划“二维转为一维的中间过程”
一维数组动态规划“最终版”
**/
// d(i, s) : 是否存在：nums区间[0, i] 中取一些元素，使其和为s
// d(i, s) = d(i-1, s){不取nums[i]} || d(i-1, s-nums[i]){取nums[i]}
// max(i) = nums.size()-1
// max(s) = sum(nums)/2
刚开始动态规划不太理解，后来发现：
我们求dp第i行的时候dp[i][?]，我们只需要知道dp的i-1行即可dp[i-1][?]。
也就是说，按照这个依赖关系，一直往下递推，只要得到第0行即可。
/*而第0行非常容易求。dp[0][s] = true当且仅当nums[0]==s*/
//图解：
//     s0 s1 s2 ...              ...sum 
// i-1 [  {s-nums[i]}  ...       s    ]
//   i [               ...       s    ]
//dp[i][s] = dp[i-1][s] || dp[i-1][s-nums[i]]
//这里要保证下标i-1>=0，所以第0行可以单独计算。
//计算方法：i==0时，s用j遍历[0, sum(nums)]区间
//发现nums[0]==s[j]，则dp[0][j]=true;
class Solution {
public:
  bool canPartition(vector<int>& nums) {
    int sum = 0;
    for(int e : nums) sum += e;
    if(sum & 1) return false;//奇数显然不符合条件
    vector<vector<bool>> d(nums.size(), vector<bool>((sum>>=1)+1, false));//sum/=2
    for(int i = 0 ; i < nums.size() ; i++){
      for(int s = 0 ; s <= sum ; s++){//s range [0, sum(nums)>>1]
        if(!i) d[i][s] = (nums[i]==s);//i==0要单独求{ nums[0]一个元素和为s }
        else d[i][s] = d[i-1][s] || (s-nums[i]>=0 ? d[i-1][s-nums[i]] : false);
      }
    }
    return d[nums.size()-1][sum];//[0,nums.size()-1]区间和为sum
  }
};
优化版本：
上面看到，我们求解dp第i行dp[i][?]的时候，只需要知道第i-1行dp[i-1][?]的值即可。
也就是说，我们没必要开这么大的二维数组空间，直接开一个一维数组空间保存前一行的值就ok了。
下面给出二维转一维的中间过程的代码。在最后会给出清晰的最终代码
class Solution {
public:
  bool canPartition(vector<int>& nums) {
    int sum = 0;
    for(int e : nums) sum += e;
    if(sum & 1) return false;
    vector<bool> d((sum>>=1)+1, false);//sum/=2
    for(int i = 0 ; i < nums.size() ; i++){
      for(int s = sum ; s >= 0 ; s--/*int s = 0 ; s <= sum ; s++*/){//从后往前，因为前面的元素我们已经求过了(i>0时确实已经求过了，i==0时我们求一遍即可，下面的代码也确实给出了i==0的情况)，可以直接用
        if(!i) d/*[i]*/[s] = (nums[i]==s);//i==0要单独求{ nums[0]一个元素和为s }
        else d/*[i]*/[s] = d/*[i-1]*/[s] || (s-nums[i]>=0 ? d/*[i-1]*/[s-nums[i]] : false);
      }
    }
    return d/*[nums.size()-1]*/[sum];//[0,nums.size()-1]区间和为sum
  }
};
/*最后，这里给出最简的一维数组动态规划代码*/
class Solution {
public:
  bool canPartition(vector<int>& nums) {
    int sum = 0;
    for(int e : nums) sum += e;
    if(sum & 1) return false;
    vector<bool> d((sum>>=1)+1, false);//sum/=2
    for(int i = 0 ; i < nums.size() ; i++){
      for(int s = sum ; s >= nums[i] ; s--){//从后往前，因为前面的元素我们已经求过了(i>0时确实已经求过了，i==0时我们求一遍即可，下面的代码也确实给出了i==0的情况)，可以直接用
        if(!i) d[s] = (nums[i]==s);//i==0要单独求{ nums[0]一个元素和为s }
        else d[s] = d[s] || d[s-nums[i]];
      }
    }
    return d[sum];
  }
};
```



```
/**
 * @param {number[]} nums
 * @return {boolean}
 * 背包问题：看数组中是否存在加起来为sum/2的数.
 * 背包容量（V） = sum/2
 * 每一个物品只能装一次,如果出现背包中重量等于sum/2则为true else false
 * dp[i]表示能否填满容量为i的背包
 * 状态转移方程为 dp[i] = dp[i-1] || nums[i]+dp[i-nums[j]]
 * 举例:v = sum/2 = 11   nums = [1,5,11,5]  1是true 0 是false
 *          0  1  2  3  4  5  6  7  8  9  10  11
 *  nums[0] 0  1  0  0  0  0  0  0  0  0   0   0
 *  nums[1] 0  1  0  0  0  1  1  0  0  0   0   0
 *  nums[2] 0  1  0  0  0  1  1  0  0  0   0   1
 *  nums[3] 0  1  0  0  0  1  1  0  0  0   0   1
 * 
 * 所以返回true，因为背包中nums[i]的状态都是由上一行决定的因此可以将二维转化为1维数组从尾部开始
 */

public boolean canPartition(int[] nums) {
    int sum = computeArraySum(nums);
    if (sum % 2 != 0) {
        return false;
    }
    int W = sum / 2;
    boolean[] dp = new boolean[W + 1];
    dp[0] = true;
    for (int num : nums) {                 // 0-1 背包一个物品只能用一次
        for (int i = W; i >= num; i--) {   // 从后往前，先计算 dp[i] 再计算 dp[i-num]
            dp[i] = dp[i] || dp[i - num];
        }
    }
    return dp[W];
}

private int computeArraySum(int[] nums) {
    int sum = 0;
    for (int num : nums) {
        sum += num;
    }
    return sum;
}
```


#[494. Target Sum (Medium)](https://leetcode.com/problems/target-sum/description/)

```
Input: nums is [1, 1, 1, 1, 1], S is 3.
Output: 5
Explanation:

-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3

There are 5 ways to assign symbols to make the sum of nums be target 3.
```

该问题可以转换为 Subset Sum 问题，从而使用 0-1 背包的方法来求解。

可以将这组数看成两部分，P 和 N，其中 P 使用正号，N 使用负号，有以下推导：

```
                  sum(P) - sum(N) = target
sum(P) + sum(N) + sum(P) - sum(N) = target + sum(P) + sum(N)
                       2 * sum(P) = target + sum(nums)
```

因此只要找到一个子集，令它们都取正号，并且和等于 (target + sum(nums))/2，就证明存在解。

```
public int findTargetSumWays(int[] nums, int S) {
    int sum = computeArraySum(nums);
    if (sum < S || (sum + S) % 2 == 1) {
        return 0;
    }
    int W = (sum + S) / 2;
    int[] dp = new int[W + 1];
    dp[0] = 1;
    for (int num : nums) {
        for (int i = W; i >= num; i--) {
            dp[i] = dp[i] + dp[i - num];
        }
    }
    return dp[W];
}

private int computeArraySum(int[] nums) {
    int sum = 0;
    for (int num : nums) {
        sum += num;
    }
    return sum;
}
```



#[474. Ones and Zeroes (Medium)](https://leetcode.com/problems/ones-and-zeroes/description/)

```
Input: Array = {"10", "0001", "111001", "1", "0"}, m = 5, n = 3
Output: 4

Explanation: There are totally 4 strings can be formed by the using of 5 0s and 3 1s, which are "10","0001","1","0"
```

这是一个多维费用的 0-1 背包问题，有两个背包大小，0 的数量和 1 的数量。

```
public int findMaxForm(String[] strs, int m, int n) {
    if (strs == null || strs.length == 0) {
        return 0;
    }
    int[][] dp = new int[m + 1][n + 1];
    for (String s : strs) {    // 每个字符串只能用一次
        int ones = 0, zeros = 0;
        for (char c : s.toCharArray()) {
            if (c == '0') {
                zeros++;
            } else {
                ones++;
            }
        }
        for (int i = m; i >= zeros; i--) {
            for (int j = n; j >= ones; j--) {
                dp[i][j] = Math.max(dp[i][j], dp[i - zeros][j - ones] + 1);
            }
        }
    }
    return dp[m][n];
}
```





#[322. Coin Change (Medium)](https://leetcode.com/problems/coin-change/description/)

```
Example 1:
coins = [1, 2, 5], amount = 11
return 3 (11 = 5 + 5 + 1)

Example 2:
coins = [2], amount = 3
return -1.
```

题目描述：给一些面额的硬币，要求用这些硬币来组成给定面额的钱数，并且使得硬币数量最少。硬币可以重复使用。

- 物品：硬币

- 物品大小：面额

- 物品价值：数量

    我们维护一个一维动态数组dp，其中dp[i]表示钱数为i时的最小硬币数的找零，注意由于数组是从0开始的，所以我们要多申请一位，数组大小为amount+1，这样最终结果就可以保存在dp[amount]中了。初始化dp[0] = 0，因为目标值若为0时，就不需要硬币了。其他值可以初始化为整型最大值，或者是amount+1，为啥呢，因为最小的硬币是1，所以amount最多需要amount个硬币，amount+1也就相当于整型最大值的作用了。好，接下来就是要找状态转移方程了，没思路？不要紧！回归例子1，假设我取了一个值为5的硬币，那么由于目标值是11，所以是不是假如我们知道dp[6]，那么就知道了组成11的dp值了？所以我们更新dp[i]的方法就是遍历每个硬币，如果遍历到的硬币值小于i值（比如我们不能用值为5的硬币去更新dp[3]）时，我们用 dp[i - coins[j]] + 1 来更新dp[i]，所以状态转移方程为：

  dp[i] = min(dp[i], dp[i - coins[j]] + 1);

  其中coins[j]为第j个硬币，而i - coins[j]为钱数i减去其中一个硬币的值，剩余的钱数在dp数组中找到值，然后加1和当前dp数组中的值做比较，取较小的那个更新dp数组。

  ```
  public int coinChange(int[] coins, int amount) {
      if(coins == null || coins.length == 0 || amount <= 0) return 0;
      int[] dp = new int[amount + 1];
      Arrays.fill(dp, amount + 1);
      dp[0] = 0;
      for(int i = 1; i <= amount; i++){
          for(int j = 0; j < coins.length; j++){
              if(coins[j] <= i){
                   dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
              }
          }
      }
      return (dp[amount] > amount) ? -1 : dp[amount];
  }
  ```



# [518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。 

 

示例 1:

输入: amount = 5, coins = [1, 2, 5]
输出: 4
解释: 有四种方式可以凑成总金额:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
示例 2:

输入: amount = 3, coins = [2]
输出: 0
解释: 只用面额2的硬币不能凑成总金额3。
示例 3:

输入: amount = 10, coins = [10] 
输出: 1


注意:

你可以假设：

0 <= amount (总金额) <= 5000
1 <= coin (硬币面额) <= 5000
硬币种类不超过 500 种
结果符合 32 位符号整数



```java
public int change(int amount, int[] coins) {
       if(amount==0)
           return 1;
        if(coins == null || coins.length == 0 ) return 0;
            int[] dp = new int[amount + 1];
            dp[0] = 1;
            for(int j = 0; j < coins.length; j++){
                for(int i = 1; i <= amount; i++){
                    if(coins[j] <= i){
                         dp[i] =dp[i]+ dp[i - coins[j]] ;
                    }
                }
            }
   		 return  dp[amount];
    }
}
```



#[139. Word Break (Medium)](https://leetcode.com/problems/word-break/description/)

```
s = "leetcode",
dict = ["leet", "code"].
Return true because "leetcode" can be segmented as "leet code".
```

dict 中的单词没有使用次数的限制，因此这是一个完全背包问题。该问题涉及到字典中单词的使用顺序，因此可理解为涉及顺序的完全背包问题。

求解顺序的完全背包问题时，对物品的迭代应该放在最里层。

```java
 public boolean wordBreak(String s, List<String> wordDict) {
        int n=s.length();
        boolean[] dp=new boolean[n+1];
        dp[0]=true;
        for(int i=1;i<=n;i++){
            for(String word:wordDict){
                int len=word.length();
                if(len<=i&&word.equals(s.substring(i-len,i)))
                    dp[i]=dp[i]||dp[i-len];
            }
        }
        return dp[n] ;
    }
```



# [309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
示例:

输入: [1,2,3,0,2]
输出: 3 
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]

```java
每次 sell 之后要等一天才能继续交易。只要把这个特点融入上一题的状态转移方程即可：

dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
dp[i][1] = max(dp[i-1][1], dp[i-2][0] - prices[i])
解释：第 i 天选择 buy 的时候，要从 i-2 的状态转移，而不是 i-1 。
翻译成代码：

int maxProfit_with_cool(int[] prices) {
    int n = prices.length;
    int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
    int dp_pre_0 = 0; // 代表 dp[i-2][0]
    for (int i = 0; i < n; i++) {
        int temp = dp_i_0;
        dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
        dp_i_1 = Math.max(dp_i_1, dp_pre_0 - prices[i]);
        dp_pre_0 = temp;
    }
    return dp_i_0;
}

```

[股票问题](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/solution/yi-ge-fang-fa-tuan-mie-6-dao-gu-piao-wen-ti-by-l-2/> /)



# [188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。

注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

示例 1:

输入: [2,4,1], k = 2
输出: 2
解释: 在第 1 天 (股票价格 = 2) 的时候买入，在第 2 天 (股票价格 = 4) 的时候卖出，这笔交易所能获得利润 = 4-2 = 2 。
示例 2:

输入: [3,2,6,5,0,3], k = 2
输出: 7
解释: 在第 2 天 (股票价格 = 2) 的时候买入，在第 3 天 (股票价格 = 6) 的时候卖出, 这笔交易所能获得利润 = 6-2 = 4 。
     随后，在第 5 天 (股票价格 = 0) 的时候买入，在第 6 天 (股票价格 = 3) 的时候卖出, 这笔交易所能获得利润 = 3-0 = 3 。



```java
 public int maxProfit(int k, int[] prices) {
        int n=prices.length;
        int[][][] dp=new int[n][k+1][2];
        if(k>n/2)
            return maxProfit_k_inf(prices);
        for(int i=0;i<n;i++){
            for(int j=k;j>=1;j--){
                if(i-1==-1){
                    dp[i][j][0]=0;
                    dp[i][j][1]=-prices[i];
                    continue;
                    
                }
                dp[i][j][0]=Math.max(dp[i-1][j][0],dp[i-1][j][1]+prices[i]);
                dp[i][j][1]=Math.max(dp[i-1][j][1],dp[i-1][j-1][0]-prices[i]);
            }
        }
        return dp[n-1][k][0];
    }
    
    
   private int maxProfit_k_inf(int[] prices) {
    int n = prices.length;
    int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
    for (int i = 0; i < n; i++) {
        int temp = dp_i_0;
        dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
        dp_i_1 = Math.max(dp_i_1, temp - prices[i]);
    }
    return dp_i_0;
}

```

该方法会超出内存限制



```java
 public int maxProfit(int k, int[] prices) {
        int n=prices.length;
        if(k>=n/2){
            int max=0;
            for(int i=1;i<n;i++){
                if(prices[i]>prices[i-1])
                    max+=prices[i]-prices[i-1];
            }
            return max;
        }
        
        int[][] max=new int[k+1][n];
        for(int i=1;i<=k;i++){
            int localMax=max[i-1][0]-prices[0];
            for(int j=1;j<n;j++){
                max[i][j]=Math.max(max[i][j-1],prices[j]+localMax);
                localMax=Math.max(localMax,max[i-1][j]-prices[j]);
            }
        }
        return max[k][n-1];
```





#[583. Delete Operation for Two Strings (Medium)](https://leetcode.com/problems/delete-operation-for-two-strings/description/)

```
Input: "sea", "eat"
Output: 2
Explanation: You need one step to make "sea" to "ea" and another step to make "eat" to "ea".
```

可以转换为求两个字符串的最长公共子序列问题。

```
public int minDistance(String word1, String word2) {
    int m = word1.length(), n = word2.length();
    int[][] dp = new int[m + 1][n + 1];
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
            }
        }
    }
    return m + n - 2 * dp[m][n];
}
```



#[72. Edit Distance (Hard)](https://leetcode.com/problems/edit-distance/description/)

```
Example 1:

Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation:
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
Example 2:

Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation:
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')
```

[题解](https://leetcode-cn.com/problems/edit-distance/solution/bian-ji-ju-chi-by-leetcode/ )

```
1.如果str1的第i个，也就是str1[i-1]和str2的第j个也就是str2[j-1]相等的话，那么

dis[i][j] = dis[i-1][j-1]

2.如果str[i-1] != str2[j-1]

　　2.1 通过替换操作把str[i-1]替换成str2[j-1]，那么

　　　　dis[i][j] = dis[i-1][j-1] + 1;

　　2.2 通过插入操作在str1后面插入str2[j-1], 那么就相当于计算

　　　　dis[i][j] = dis[i][j-1] + 1;

　　2.3 通过插入操作在str2后面插入str1[i-1],那么就是

　　　　dis[i][j] = dis[i-1][j] + 1;　　

　　在上述三个中选一个最小的。迭代更新。
```



```java
public int minDistance(String word1, String word2) {
    int n = word1.length();
    int m = word2.length();

    // if one of the strings is empty
    if (n * m == 0)
      return n + m;

    // array to store the convertion history
    int [][] d = new int[n + 1][m + 1];

    // init boundaries
    for (int i = 0; i < n + 1; i++) {
      d[i][0] = i;
    }
    for (int j = 0; j < m + 1; j++) {
      d[0][j] = j;
    }

    // DP compute 
    for (int i = 1; i < n + 1; i++) {
      for (int j = 1; j < m + 1; j++) {
        int left = d[i - 1][j] + 1;
        int down = d[i][j - 1] + 1;
        int left_down = d[i - 1][j - 1];
        if (word1.charAt(i - 1) != word2.charAt(j - 1))
          left_down += 1;
        d[i][j] = Math.min(left, Math.min(down, left_down));//注意此处Math.min函数的用法

      }
    }
    return d[n][m];
  }

```

