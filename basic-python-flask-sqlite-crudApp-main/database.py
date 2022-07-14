import sqlite3
import datetime

def getConnection():
    conn = sqlite3.connect("car.db")
    conn.row_factory = sqlite3.Row
    return conn

def create():
    conn = getConnection()

    try:
        query = """
    create table company(id integer primary key autoincrement, 
    name varchar(30) not null, 
    address text, 
    phone varchar(10))
    """
        
        conn.execute(query)
        conn.commit()

        query = """
    create table car(id integer primary key autoincrement, 
    name varchar(30) not null, 
    model varchar(30) not null, 
    color varchar(10) default "red", 
    launch_date date default CURRENT_DATE,
    company integer not null,
    foreign key (company) references company(id))
    """
        conn.execute(query)
        conn.commit()
    except Exception as e:
        print(e)
    finally:
        conn.close()


def addCompany(name, address, phone):
    conn = getConnection()

    try:
        query = """
                insert into company(name, address, phone) values (?,?,?)
                """
        conn.execute(query,[name, address, phone])
        conn.commit()
        print("SUCCESS!!")
    except Exception as e:
        print(e)

    finally:
        conn.close()
 
def getCompany(name):
    conn = getConnection()

    try:
        query = """
                select * from company where name = ?
                """
        cursor = conn.execute(query,[name])
        print(cursor.fetchall())
        conn.commit()
        print("SUCCESS!!")
    except Exception as e:
        print(e)

    finally:
        conn.close()

def addCar(name, model, color, launch_date, company_id):
    conn = getConnection()

    try:
        query = """
                insert into car(name, model, color, launch_date, company)  values (?,?,?,?,?)
                """
        conn.execute(query,[name, model, color, launch_date, company_id])
        
        conn.commit()
        return "SUCCESS!!"
    except Exception as e:
        print(e)
        return "FAILURE "+e

    finally:
        conn.close()

def getCars():
    conn = getConnection()

    try:
        query = """
                select car.name, car.model, car.color, car.launch_date, company.name as company from car inner join company on car.company = company.id
                """
        cursor = conn.execute(query)
        return [dict(car) for car in cursor.fetchall()]            
    except Exception as e:
        print(e)

    finally:
        conn.close()

def getCompanies():
    conn = getConnection()

    try:
        query = """
                select * from  company
                """
        cursor = conn.execute(query)
        return [dict(car) for car in cursor.fetchall()]
            
        print("SUCCESS!!")
    except Exception as e:
        print(e)

    finally:
        conn.close()


if __name__ == "__main__":
    # create()
    # addCompany("tata","gujrat","9888998899")
    # getCompany("tata")
    # addCar("sumo","V3","white", datetime.date(2000,1,10),2)
    getCars()