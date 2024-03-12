#%%
import sqlite3
from datetime import datetime
from abc import ABC, abstractclassmethod

class DatabaseTable(ABC):
    def __init__(self, db_path, table_name):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.table_name = table_name
        self.create_table()

    @abstractclassmethod
    def create_table(self):
        """
        Abstract method to create a table in the database.
        
        This method should be implemented by subclasses to define the schema and create a table in the database according to the specific requirements of the subclass.

        Returns:
            None

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Notes:
            This method should be overridden by subclasses to provide the necessary SQL statements to create a table in the database. It should handle the creation of the table schema and any necessary constraints.

        Example:
            This method should be implemented in a subclass with the specific SQL statements required to create a table in the database. For example:

            >>> class MyDatabaseHandler(DatabaseHandler):
            >>>     def create_table(self):
            >>>         sql_statement = "CREATE TABLE IF NOT EXISTS my_table (id INTEGER PRIMARY KEY, name TEXT)"
            >>>         self.cursor.execute(sql_statement)
            >>>         self.connection.commit()

        """
        pass

    def __validate_columns_names(self, columns_names):
        """
                Validates the column names against the existing database columns.

        This method checks whether the provided column names are valid by comparing them against the column names existing in the database. It raises an exception if any of the provided column names are not found in the database.

        Args:
            columns_names (list): A list of column names to validate.

        Returns:
            None

        Raises:
            Exception: If any of the provided column names are not found in the database.
        
        Notes:
            This method is intended to be used internally within other methods to validate column names before performing database operations. It ensures that only valid column names are used in database queries and prevents potential errors due to invalid column names.

        Example:
            >>> obj = MyClass()
            >>> columns = ['column1', 'column2']
            >>> obj.__validate_columns_names(columns)
        """
        db_columns_names = self.get_columns_names()
        for col in columns_names:
            if col not in db_columns_names:
                raise Exception(f'Invalid column name: {col}')
    
    def get_columns_names(self):
        """
        Retrieves the names of columns in the database table.

        This method queries the SQLite database to retrieve information about the columns in the specified table. It returns a list containing the names of all columns in the table.

        Returns:
            list: A list containing the names of columns in the database table.

        Notes:
            This method executes a PRAGMA statement to retrieve information about the columns in the database table. It then extracts and returns the column names from the obtained column information.

        Example:
            To retrieve the names of columns in the database table, you can use this method as follows:

            >>> obj = MyClass()
            >>> columns_names = obj.get_columns_names()
            >>> print(columns_names)
        """
        self.cursor.execute(f"PRAGMA table_info({self.table_name})")
        column_info = self.cursor.fetchall()
        return [col[1] for col in column_info]

    def row_to_dict(self, row):
        """
        Converts a database row to a dictionary with column names as keys.

        This method takes a database row, typically obtained from a database query result, and converts it into a dictionary where the keys are column names and the values are the corresponding values from the row. It uses the column names obtained from the database table schema to ensure that the dictionary keys match the column names.

        Args:
            row (tuple): A database row obtained from a query result.

        Returns:
            dict: A dictionary where keys are column names and values are the corresponding values from the row.

        Notes:
            This method is useful for converting database query results into a more accessible format for further processing or manipulation in Python code. It helps provide a convenient interface between the database and the application code.

        Example:
            >>> obj = MyClass()
            >>> row = (1, 'John Doe', 30)
            >>> result_dict = obj.row_to_dict(row)
            >>> print(result_dict)
            {'id': 1, 'name': 'John Doe', 'age': 30}
        """
        keys = self.get_columns_names()
        row_dict = {key:row[i] for i, key in enumerate(keys)}
        return row_dict
    
    def insert_data(self, **options):
        """
        Inserts data into the database table.

        This method inserts data into the database table specified by the class instance. It allows for insertion of data into specific columns by passing keyword arguments where the keys are column names and the values are the corresponding values to be inserted into those columns. The method also automatically adds the current date and time to a 'date' column in the table.

        Args:
            **options (dict): Keyword arguments representing column names and their corresponding values to be inserted into the database.

        Returns:
            None

        Raises:
            Exception: If any of the specified column names are invalid.

        Notes:
            This method provides a convenient way to insert data into a database table by allowing the user to specify the column names and values as keyword arguments. It automatically adds the current date and time to a 'date' column in the table.

        Example:
            >>> obj = MyClass()
            >>> obj.insert_data(name='John Doe', age=30, city='New York')
        """
        columns_names = list(options.keys())
        self.__validate_columns_names(columns_names)
        values = list(options.values())
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        columns_names.append('date')
        values.append(current_date)

        columns_query = f"({','.join([col for col in columns_names])})"
        values_slots = f"{'?,'*len(values)}"[:-1]

        query = f"INSERT INTO {self.table_name} {columns_query} VALUES ({values_slots})"

        self.cursor.execute(query, values)
        self.conn.commit()
    
    def get(self, **conditions):
        """
        Retrieves a single row from the database table based on the specified condition.

        This method retrieves a single row from the database table specified by the class instance based on the specified condition. The condition is specified as a keyword argument where the key represents the column name and the value represents the value to match in that column. If a row matching the condition is found, it is returned as a dictionary where the keys are column names and the values are the corresponding values from the row. If no matching row is found, None is returned.

        Args:
            **option (dict): Keyword argument representing the condition to match in the database query.

        Returns:
            dict or None: A dictionary representing a single row from the database table if found, otherwise None.

        Raises:
            Exception: If more than one condition is specified.

        Notes:
            This method provides a simple way to retrieve a single row from a database table based on a specified condition. It uses a single condition specified as a keyword argument where the key represents the column name and the value represents the value to match in that column.

        Example:
            >>> obj = MyClass()
            >>> row = obj.get(id=1)
            >>> print(row)
        """
        columns = list(conditions.keys())
        values = list(conditions.values())
        self.__validate_columns_names(columns)

        conditions_query = ' AND '.join([f"{col} = ?" for col in columns])
        query = f"SELECT * FROM {self.table_name} WHERE {conditions_query}"

        self.cursor.execute(query, values)
        rows = self.cursor.fetchall()
        rows_dict = [self.row_to_dict(r) for r in rows]
        return rows_dict

    def remove(self, **conditions):
        """
        Removes rows from the database table based on the specified condition.

        This method removes rows from the database table specified by the class instance based on the specified condition. The condition is specified as a keyword argument where the key represents the column name and the value represents the value to match in that column.

        Args:
            **option (dict): Keyword argument representing the condition to match in the database query.

        Returns:
            None

        Raises:
            Exception: If more than one condition is specified.

        Notes:
            This method provides a convenient way to remove rows from a database table based on a specified condition. It uses a single condition specified as a keyword argument where the key represents the column name and the value represents the value to match in that column.

        Example:
            >>> obj = MyClass()
            >>> obj.remove(id=1)
        """
        columns = list(conditions.keys())
        values = list(conditions.values())
        self.__validate_columns_names(columns)

        conditions_query = ' AND '.join([f"{col} = ?" for col in columns])
        query = f"DELETE FROM {self.table_name} WHERE {conditions_query}"

        self.cursor.execute(query, values)
        self.conn.commit()

    def get_all(self):
        """
        Retrieves all rows from the database table.

        This method executes a SQL query to retrieve all rows from the database table specified by the class instance. It then converts the retrieved rows into dictionaries, where the keys represent the column names and the values represent the corresponding values in the row.

        Returns:
            list: A list of dictionaries representing the rows retrieved from the database table.

        Notes:
            This method provides a convenient way to retrieve all rows from a database table. It executes a simple SQL query to select all rows from the table and then converts the result set into a list of dictionaries for easier manipulation and processing.

        Example:
            >>> obj = MyClass()
            >>> all_rows = obj.get_all()
            >>> print(all_rows)
        """
        query = f"SELECT * FROM {self.table_name}"
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        rows_dict = [self.row_to_dict(r) for r in rows]
        return rows_dict

    def close(self):
        """
        Closes the database connection and cursor.

        This method closes the database connection and cursor associated with the class instance. It is recommended to call this method when finished with the database operations to release any resources held by the connection and cursor.

        Returns:
            None

        Notes:
            Closing the database connection and cursor helps to release any resources held by them and ensures proper cleanup after database operations. It is good practice to call this method when the database operations are completed to prevent resource leaks.

        Example:
            >>> obj = MyClass()
            >>> obj.close()
        """
        self.cursor.close()
        self.conn.close()

class ScrapingDatabase(DatabaseTable):
    def __init__(self, db_path, table_name):
        super().__init__(db_path, table_name)
    
    def create_table(self):
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.table_name} (
            link_pdf TEXT PRIMARY KEY NOT NULL,
            agency TEXT NOT NULL,
            created_at TEXT,
            date TEXT
        )''')

        self.conn.commit()

class EditalDatabse(DatabaseTable):
    def __init__(self, db_path, table_name):
        super().__init__(db_path, table_name)

    def create_table(self):
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.table_name} (
            link_pdf TEXT PRIMARY KEY,
            agency TEXT,
            titulo TEXT,
            objetivo TEXT,
            elegibilidade TEXT,
            submissao TEXT,
            financiamento TEXT,
            areas TEXT,
            date TEXT
        )''')

        self.conn.commit()
    
if __name__ == '__main__':
    scraping_db = ScrapingDatabase('web_data.db', 'web_data')
    editais_db = ScrapingDatabase('web_data.db', 'editais')
    # scraping_db.insert_data(agency='centelha', link_pdf='example.com', agency_link='finep.com', hash='hash123')
    print(editais_db.get_columns_names())
    editais_db.insert_data(
        agency = "centelha",
        link_pdf = "pdf.com",
        titulo = "titulo do pdf",
        objetivo = "objetivo",
        elegibilidade = "Os critérios são...",
        submissao = "10/10/10",
        financiamento = "100,00",
        areas = "computacao",
    )
    print(scraping_db.get_all())
    print(editais_db.get_all())
    scraping_db.close()
    editais_db.close()