import sqlite3

def createTables():
	query='''create table category(id integer primary key,name varchar(255) not null)'''
	conn=sqlite3.connect("data.db")
	conn.execute(query)
	conn.commit()

	query='''create table product(id integer primary key,name varchar(255) not null,category_id integer not null,foreign key(category_id) references category(id))'''
	conn.execute(query)
	conn.commit()
	conn.close()

def addCategory(name):
	query='''insert into category(name) values(?)'''
	conn=sqlite3.connect("data.db")
	try:
		conn.execute(query,[name])
		conn.commit()
		return "Added Successfully"
	except Exception as e:
		print(e)
		return "Could not add category"

	finally:
		conn.close()

def getCategory():
	query='''select *from category'''
	conn=sqlite3.connect("data.db")
	try:
		conn.execute(query)
		return conn.execute(query).fetchall()
		
	except Exception as e:
		print(e)

	finally:
		conn.close()

def addProduct(name,category_id):
	query='''insert into product(name,category_id) values (?,?)'''
	conn=sqlite3.connect("data.db")
	try:
		conn.execute(query,[name,category_id])
		conn.commit()
		print("Added Successfully")
	except Exception as e:
		print(e)

	finally:
		conn.close()


def getProducts():
	query='''select product.name as pname,category.name as cname from product inner join category on product.category_id=category.id'''
	conn=sqlite3.connect("data.db")
	try:
		conn.execute(query)
		return conn.execute(query).fetchall()
		
	except Exception as e:
		print(e)

	finally:
		conn.close()
	

if __name__=='__main__':
	createTables()
	addCategory("Dairy")
	addCategory("Cerals")
	print(getCategory())
	addProduct("Milk",1)
	addProduct("Pedha",1)
	addProduct("Jowar",2)
	print(getProducts())

	