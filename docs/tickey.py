def count_combinations(N, K, pieces):
    dp = [0] * (K + 1)
    dp[0] = 1

    for piece in pieces:
        for i in range(K, piece - 1, -1):
            dp[i] += dp[i - piece]

    return dp[K]

N, K = map(int, input().split())
pieces = list(map(int, input().split()))

combinations = count_combinations(N, K, pieces)
print(combinations)
