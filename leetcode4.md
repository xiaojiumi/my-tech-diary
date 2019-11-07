# 1. 字符串循环移位包含

[编程之美 3.1](https://github.com/CyC2018/CS-Notes/blob/master/notes/Leetcode 题解 - 字符串.md#)

```
s1 = AABCD, s2 = CDAA
Return : true
```

给定两个字符串 s1 和 s2，要求判定 s2 是否能够被 s1 做循环移位得到的字符串包含。

s1 进行循环移位的结果是 s1s1 的子字符串，因此只要判断 s2 是否是 s1s1 的子字符串即可。

```java
判断子串 
s1.indexOf(s2);
返回值为-1说明不是
```



# 2. 字符串循环移位

[编程之美 2.17](https://github.com/CyC2018/CS-Notes/blob/master/notes/Leetcode 题解 - 字符串.md#)

```
s = "abcd123" k = 3
Return "123abcd"
```

将字符串向右循环移动 k 位。

将 abcd123 中的 abcd 和 123 单独翻转，得到 dcba321，然后对整个字符串进行翻转，得到 123abcd。

```java
字符串翻转
public String reverse(String s){
		List<Character> l=new ArrayList<>();
        for(char c:s.toCharArray()){
            l.add(c);
        }
        Collections.reverse(l);
        StringBuilder sb=new StringBuilder();
        for(char c:l){
            sb.append(c);
        }
        return sb.toString();
}
```



# 3. 字符串中单词的翻转

[程序员代码面试指南](https://github.com/CyC2018/CS-Notes/blob/master/notes/Leetcode 题解 - 字符串.md#)

```
s = "I am a student"
Return "student a am I"
```

将每个单词翻转，然后将整个字符串翻转。

# 5. 计算一组字符集合可以组成的回文字符串的最大长度

\409. Longest Palindrome (Easy)

[Leetcode](https://leetcode.com/problems/longest-palindrome/description/) / [力扣](https://leetcode-cn.com/problems/longest-palindrome/description/)

```
Input : "abccccdd"
Output : 7
Explanation : One longest palindrome that can be built is "dccaccd", whose length is 7.
```

使用长度为 256 的整型数组来统计每个字符出现的个数，每个字符有偶数个可以用来构成回文字符串。

因为回文字符串最中间的那个字符可以单独出现，所以如果有单独的字符就把它放到最中间。

```
public int longestPalindrome(String s) {
    int[] cnts = new int[256];
    for (char c : s.toCharArray()) {
        cnts[c]++;
    }
    int palindrome = 0;
    for (int cnt : cnts) {
        palindrome += (cnt / 2) * 2;
    }
    if (palindrome < s.length()) {
        palindrome++;   // 这个条件下 s 中一定有单个未使用的字符存在，可以把这个字符放到回文的最中间
    }
    return palindrome;
}
```

# 6. 字符串同构

\205. Isomorphic Strings (Easy)

[Leetcode](https://leetcode.com/problems/isomorphic-strings/description/) / [力扣](https://leetcode-cn.com/problems/isomorphic-strings/description/)

```
Given "egg", "add", return true.
Given "foo", "bar", return false.
Given "paper", "title", return true.
```

记录一个字符上次出现的位置，如果两个字符串中的字符上次出现的位置一样，那么就属于同构。

```
public boolean isIsomorphic(String s, String t) {
    int[] preIndexOfS = new int[256];
    int[] preIndexOfT = new int[256];
    for (int i = 0; i < s.length(); i++) {
        char sc = s.charAt(i), tc = t.charAt(i);
        if (preIndexOfS[sc] != preIndexOfT[tc]) {
            return false;
        }
        preIndexOfS[sc] = i + 1;
        preIndexOfT[tc] = i + 1;
    }
    return true;
}
```

# 7. 回文子字符串个数

\647. Palindromic Substrings (Medium)

[Leetcode](https://leetcode.com/problems/palindromic-substrings/description/) / [力扣](https://leetcode-cn.com/problems/palindromic-substrings/description/)

```
Input: "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
```

从字符串的某一位开始，尝试着去扩展子字符串。

```
private int cnt = 0;

public int countSubstrings(String s) {
    for (int i = 0; i < s.length(); i++) {
        extendSubstrings(s, i, i);     // 奇数长度
        extendSubstrings(s, i, i + 1); // 偶数长度
    }
    return cnt;
}

private void extendSubstrings(String s, int start, int end) {
    while (start >= 0 && end < s.length() && s.charAt(start) == s.charAt(end)) {
        start--;
        end++;
        cnt++;
    }
}
```

# 8. 判断一个整数是否是回文数

\9. Palindrome Number (Easy)

[Leetcode](https://leetcode.com/problems/palindrome-number/description/) / [力扣](https://leetcode-cn.com/problems/palindrome-number/description/)

要求不能使用额外空间，也就不能将整数转换为字符串进行判断。

将整数分成左右两部分，右边那部分需要转置，然后判断这两部分是否相等。

```java
public class Solution {
    public bool IsPalindrome(int x) {
        // 特殊情况：
        // 如上所述，当 x < 0 时，x 不是回文数。
        // 同样地，如果数字的最后一位是 0，为了使该数字为回文，
        // 则其第一位数字也应该是 0
        // 只有 0 满足这一属性
        if(x < 0 || (x % 10 == 0 && x != 0)) {
            return false;
        }

        int revertedNumber = 0;
        while(x > revertedNumber) {
            revertedNumber = revertedNumber * 10 + x % 10;
            x /= 10;
        }

        // 当数字长度为奇数时，我们可以通过 revertedNumber/10 去除处于中位的数字。
        // 例如，当输入为 12321 时，在 while 循环的末尾我们可以得到 x = 12，revertedNumber = 123，
        // 由于处于中位的数字不影响回文（它总是与自己相等），所以我们可以简单地将其去除。
        return x == revertedNumber || x == revertedNumber/10;
    }
}
```

# 9. 统计二进制字符串中连续 1 和连续 0 数量相同的子字符串个数

\696. Count Binary Substrings (Easy)

[Leetcode](https://leetcode.com/problems/count-binary-substrings/description/) / [力扣](https://leetcode-cn.com/problems/count-binary-substrings/description/)

```
Input: "00110011"
Output: 6
Explanation: There are 6 substrings that have equal number of consecutive 1's and 0's: "0011", "01", "1100", "10", "0011", and "01".
```

方法一：按字符分组
我们可以将字符串 s 转换为 groups 数组表示字符串中相同字符连续块的长度。例如，如果 s=“11000111000000”，则 groups=[2，3，4，6]。

对于 '0' * k + '1' * k 或 '1' * k + '0' * k 形式的每个二进制字符串，此字符串的中间部分必须出现在两个组之间。

让我们尝试计算 groups[i] 和 groups[i+1] 之间的有效二进制字符串数。如果我们有 groups[i] = 2, groups[i+1] = 3，那么它表示 “00111” 或 “11000”。显然，我们可以在此字符串中生成 min(groups[i], groups[i+1]) 有效的二进制字符串。

算法：

让我们创建上面定义的 groups。s 的第一个元素属于它自己的组。每个元素要么与前一个元素不匹配，从而开始一个大小为 1 的新组；要么匹配，从而使最近一个组的大小增加 1。
然后，我们将取 min(groups[i-1], groups[i]) 的和。

```java
class Solution {
    public int countBinarySubstrings(String s) {
        int[] groups=new int[s.length()];
        int t=0;
        groups[0]=1;
        for(int i=1;i<s.length();i++){
            if(s.charAt(i-1)!=s.charAt(i)){
                groups[++t]=1;
            }else{
                groups[t]++;
            }
        }
        int res=0;
        for(int i=1;i<=t;i++){
            res+=Math.min(groups[i-1],groups[i]);
        }
        return res;
    }
}
```

# 1. 把数组中的 0 移到末尾

\283. Move Zeroes (Easy)

[Leetcode](https://leetcode.com/problems/move-zeroes/description/) / [力扣](https://leetcode-cn.com/problems/move-zeroes/description/)

```
For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].
public void moveZeroes(int[] nums) {
    int idx = 0;
    for (int num : nums) {
        if (num != 0) {
            nums[idx++] = num;
        }
    }
    while (idx < nums.length) {
        nums[idx++] = 0;
    }
}
```

# 4. 有序矩阵查找

240. Search a 2D Matrix II (Medium)

[Leetcode](https://leetcode.com/problems/search-a-2d-matrix-ii/description/) / [力扣](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/description/)

因为矩阵的行和列是排序的（分别从左到右和从上到下），所以在查看任何特定值时，我们可以修剪O(m)O(m)或O(n)O(n)元素。

算法：
首先，我们初始化一个指向矩阵左下角的 (row，col)(row，col) 指针。然后，直到找到目标并返回 true（或者指针指向矩阵维度之外的 (row，col)(row，col) 为止，我们执行以下操作：如果当前指向的值大于目标值，则可以 “向上” 移动一行。 否则，如果当前指向的值小于目标值，则可以移动一列。不难理解为什么这样做永远不会删减正确的答案；因为行是从左到右排序的，所以我们知道当前值右侧的每个值都较大。 因此，如果当前值已经大于目标值，我们知道它右边的每个值会比较大。也可以对列进行非常类似的论证，因此这种搜索方式将始终在矩阵中找到目标（如果存在）。

这个算法的关键是找到合适的遍历起点，这个点肯定具有某种特殊性，这个二维矩阵，四个角就是四个特殊点，但他们的特点不同，左上和右下分别是矩阵的最小和最大值，左下和右上具有两面性，如果是所在行最大值那么就是所在列的最小值，反过来也一样。左上和右下与目标值比较不相等时，下一步既可以遍历行也可以遍历列是不确定的，而左下和右上是可以确定的，因为自身值的特点可以排除一个方向的路径，只有一个遍历路径。

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        if(matrix==null||matrix.length==0||matrix[0].length==0)return false;
        int m=matrix.length,n=matrix[0].length;
        int row=0,col=n-1;
        while(row<m&&col>=0){
            if(target==matrix[row][col])return true;
            else if(target>matrix[row][col]) row++;
            else col--;
        }
        return false;
    }
}
```

# 5. 有序矩阵的 Kth Element

\378. Kth Smallest Element in a Sorted Matrix ((Medium))

[Leetcode](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/description/) / [力扣](https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/description/)

```
matrix = [
  [ 1,  5,  9],
  [10, 11, 13],
  [12, 13, 15]
],
k = 8,

return 13.
```

��题参考：[Share my thoughts and Clean Java Code

[Leetcode](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/discuss/85173) / [力扣](https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/discuss/85173)

二分查找解法：

```
public int kthSmallest(int[][] matrix, int k) {
    int m = matrix.length, n = matrix[0].length;
    int lo = matrix[0][0], hi = matrix[m - 1][n - 1];
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int cnt = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n && matrix[i][j] <= mid; j++) {
                cnt++;
            }
        }
        if (cnt < k) lo = mid + 1;
        else hi = mid - 1;
    }
    return lo;
}
```

堆解法：

```
public int kthSmallest(int[][] matrix, int k) {
    int m = matrix.length, n = matrix[0].length;
    PriorityQueue<Tuple> pq = new PriorityQueue<Tuple>();
    for(int j = 0; j < n; j++) pq.offer(new Tuple(0, j, matrix[0][j]));
    for(int i = 0; i < k - 1; i++) { // 小根堆，去掉 k - 1 个堆顶元素，此时堆顶元素就是第 k 的数
        Tuple t = pq.poll();
        if(t.x == m - 1) continue;
        pq.offer(new Tuple(t.x + 1, t.y, matrix[t.x + 1][t.y]));
    }
    return pq.poll().val;
}

class Tuple implements Comparable<Tuple> {
    int x, y, val;
    public Tuple(int x, int y, int val) {
        this.x = x; this.y = y; this.val = val;
    }

    @Override
    public int compareTo(Tuple that) {
        return this.val - that.val;
    }
}
```

# 6. 一个数组元素在 [1, n] 之间，其中一个数被替换为另一个数，找出重复的数和丢失的数

\645. Set Mismatch (Easy)

[Leetcode](https://leetcode.com/problems/set-mismatch/description/) / [力扣](https://leetcode-cn.com/problems/set-mismatch/description/)

```
Input: nums = [1,2,2,4]
Output: [2,3]
```

从题目描述，可看出以下几点：

正确的数组，排序后其元素应该是：1，2，3 ... n（当然，此题我们不需要排序）
错误的数组，排序后其元素应该是：1，2，3 ... m，m, ... n

此时，就出现了三种情况，重复的元素，出现了 2 次，正确的元素出现了 1 次，丢失的元素出现了 0 次。

于是，很自然的可以想到，用一个数组进行存储其出现的次数。

定义一个元素出现次数的数组：
int count[] = new int[10002];

其中，count数组下标为错误的数组nums的元素值
通过遍历数组nums，利用
count[nums[i]]++;
来计算每一个元素出现的次数，次数存储在count中

最后，遍历count数组，其值为 0 的便是丢失的数组，其值为 2 的便是重复的数组



    public int[] findErrorNums(int[] nums) {
    int out[] = new int[2];
    
        //1。方法一：
        int count[] = new int[10002];
        for(int i = 0; i < nums.length; i++){
            count[nums[i]]++;
    
        }
    
        for(int i = 0; i <= nums.length; i++){
            if(count[i] == 2)
                out[0] = i;
            if(count[i] == 0 && i != 0 && i <= nums.length)
                out[1] = i;
        }
    
        return out;
    
    }

# 7. 找出数组中重复的数，数组值在 [1, n] 之间

\287. Find the Duplicate Number (Medium)

[Leetcode](https://leetcode.com/problems/find-the-duplicate-number/description/) / [力扣](https://leetcode-cn.com/problems/find-the-duplicate-number/description/)

要求不能修改数组，也不能使用额外的空间。

方法：二分法
关键：这道题的关键是对要定位的“数”做二分，而不是对数组的索引做二分。要定位的“数”根据题意在 11 和 nn 之间，每一次二分都可以将搜索区间缩小一半。

以 [1, 2, 2, 3, 4, 5, 6, 7] 为例，一共有 88 个数，每个数都在 11 和 77 之间。11 和 77 的中位数是 44，遍历整个数组，统计小于 44 的整数的个数，至多应该为 33 个，如果超过 33 个就说明重复的数存在于区间 [1,4)[1,4) （注意：左闭右开）中；否则，重复的数存在于区间 [4,7][4,7]（注意：左右都是闭）中。这里小于 44 的整数有 44 个（它们是 1, 2, 2, 3），因此砍掉右半区间，连中位数也砍掉。以此类推，最后区间越来越小，直到变成 11 个整数，这个整数就是我们要找的重复的数。

```
public int findDuplicate(int[] nums) {
     int l = 1, h = nums.length - 1;
     while (l <= h) {
         int mid = l + (h - l) / 2;
         int cnt = 0;
         for (int i = 0; i < nums.length; i++) {
             if (nums[i] <= mid) cnt++;
         }
         if (cnt > mid) h = mid - 1;
         else l = mid + 1;
     }
     return l;
}
```

# 8. 数组相邻差值的个数

\667. Beautiful Arrangement II (Medium)

[Leetcode](https://leetcode.com/problems/beautiful-arrangement-ii/description/) / [力扣](https://leetcode-cn.com/problems/beautiful-arrangement-ii/description/)

```
Input: n = 3, k = 2
Output: [1, 3, 2]
Explanation: The [1, 3, 2] has three different positive integers ranging from 1 to 3, and the [2, 1] has exactly 2 distinct integers: 1 and 2.
```

题目描述：数组元素为 1~n 的整数，要求构建数组，使得相邻元素的差值不相同的个数为 k。

让前 k+1 个元素构建出 k 个不相同的差值，序列为：1 k+1 2 k 3 k-1 ... k/2 k/2+1.

找规律 https://blog.csdn.net/qq_41855420/article/details/89396571

```java
class Solution {
    public int[] constructArray(int n, int k) {
        int[] res=new int[n];
        int num1=1,num2=k+1;
        for(int i=0;i<=k;i+=2){
            res[i]=num1++;
        }
        for(int i=1;i<=k;i+=2){
            res[i]=num2--;
        }
        for(int i=k+1;i<n;i++){
            res[i]=i+1;
        }
        return res;
    }
}
```

# 9. 数组的度

\697. Degree of an Array (Easy)

[Leetcode](https://leetcode.com/problems/degree-of-an-array/description/) / [力扣](https://leetcode-cn.com/problems/degree-of-an-array/description/)

```
Input: [1,2,2,3,1,4,2]
Output: 6
```

题目描述：数组的度定义为元素出现的最高频率，例如上面的数组度为 3。要求找到一个最小的子数组，这个子数组的度和原数组一样。

算法：

具有度数 d 的数组必须有一些元素 x 出现 d 次。如果某些子数组具有相同的度数，那么某些元素 x （出现 d 次）。最短的子数组是将从 x 的第一次出现到最后一次出现的数组。
对于给定数组中的每个元素，让我们知道 left 是它第一次出现的索引； right 是它最后一次出现的索引。例如，当 nums = [1,2,3,2,5] 时，left[2] = 1 和 right[2] = 3。
然后，对于出现次数最多的每个元素 x，right[x] - left[x] + 1 将是我们的候选答案，我们将取这些候选的最小值。

```java
class Solution {
    public int findShortestSubArray(int[] nums) {
        Map<Integer, Integer> left = new HashMap(),
            right = new HashMap(), count = new HashMap();

        for (int i = 0; i < nums.length; i++) {
            int x = nums[i];
            if (left.get(x) == null) left.put(x, i);
            right.put(x, i);
            count.put(x, count.getOrDefault(x, 0) + 1);
        }

        int ans = nums.length;
        int degree = Collections.max(count.values());
        for (int x: count.keySet()) {
            if (count.get(x) == degree) {
                ans = Math.min(ans, right.get(x) - left.get(x) + 1);
            }
        }
        return ans;
    }
}
```

# 11. 嵌套数组

\565. Array Nesting (Medium)

[Leetcode](https://leetcode.com/problems/array-nesting/description/) / [力扣](https://leetcode-cn.com/problems/array-nesting/description/)

```
Input: A = [5,4,0,3,1,6,2]
Output: 4
Explanation:
A[0] = 5, A[1] = 4, A[2] = 0, A[3] = 3, A[4] = 1, A[5] = 6, A[6] = 2.

One of the longest S[K]:
S[0] = {A[0], A[5], A[6], A[2]} = {5, 6, 2, 0}
```

题目描述：S[i] 表示一个集合，集合的第一个元素是 A[i]，第二个元素是 A[A[i]]，如此嵌套下去。求最大的 S[i]。

```
public int arrayNesting(int[] nums) {
    int max = 0;
    for (int i = 0; i < nums.length; i++) {
        int cnt = 0;
        for (int j = i; nums[j] != -1; ) {
            cnt++;
            int t = nums[j];
            nums[j] = -1; // 标记该位置已经被访问
            j = t;

        }
        max = Math.max(max, cnt);
    }
    return max;
}
```

# 12. 分隔数组

\769. Max Chunks To Make Sorted (Medium)

[Leetcode](https://leetcode.com/problems/max-chunks-to-make-sorted/description/) / [力扣](https://leetcode-cn.com/problems/max-chunks-to-make-sorted/description/)

```
Input: arr = [1,0,2,3,4]
Output: 4
Explanation:
We can split into two chunks, such as [1, 0], [2, 3, 4].
However, splitting into [1, 0], [2], [3], [4] is the highest number of chunks possible.
```

题目描述：分隔数组，使得对每部分排序后数组就为有序。

```
public int maxChunksToSorted(int[] arr) {
    if (arr == null) return 0;
    int ret = 0;
    int right = arr[0];
    for (int i = 0; i < arr.length; i++) {
        right = Math.max(right, arr[i]);
        if (right == i) ret++;
    }
    return ret;
}
```

