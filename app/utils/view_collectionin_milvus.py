from pymilvus import connections, utility

connections.connect("default", host="localhost", port="19530")
print("Collections:", utility.list_collections())
