import redis
r = redis.from_url("redis://127.0.0.1:6379/0", decode_responses=True)
print(r.ping())