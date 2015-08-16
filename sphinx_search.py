import pymysql
import pymysql.cursors
import pymongo
from bson import json_util, ObjectId


SPHINXSEARCH_HOST = '172.16.4.52'
SPHINXQL_PORT = 9306
SPHINX_LIMIT = 10000000

def sphinx_query(index_name, query="", facet=None, limit=None):
    """
    Search Sphinx index using a simple match via SphinxQL
    :param index_name: Name of the index to search on
    :param query: String with the query expression
    :param facet: Attribute name to facet by. Must be a list
    :return: JSON (array of objects)
    """

    # Setup Sphinxsearch SphinxQL connection
    sphinx_conn = pymysql.connect(host=SPHINXSEARCH_HOST, port=SPHINXQL_PORT)
    cursor = sphinx_conn.cursor(pymysql.cursors.DictCursor)

    if facet is None:
        if index_name == 'mediacloud_tweets':
            order_field = 'created_at_datetime'
        else:
            order_field = 'published'
        # We're concatenating strings here because DB API substitution only
        # seem to work for values, not tables or fields. Since these values are
        # white-listed here (they are not influenced by user's input) this
        # should not put us in risk.
        cursor.execute("SELECT * from " + index_name + " WHERE MATCH(%(query)s) "
                "ORDER BY " + order_field + " DESC "
                "LIMIT %(limit)s OPTION max_matches=%(limit)s",
                    {'query': query, 'limit': limit or SPHINX_LIMIT})
    else:
        cursor.execute("SELECT * from "+index_name+" WHERE MATCH(%s) " + " ".join(["FACET {}".format(f) for f in facet]),
                       (query,))
    results = cursor.fetchall()
    cursor.close()
    return results


def fetch_docs(ids, collection):
    client = pymongo.Connection('dirrj',27017)
    db = client.MCDB
    col = db[collection]
    cur = col.find({"_id": {"$in": ids}})
    return cur


if __name__ == '__main__':
    query = 'Charlie Hebdo'
    index = 'mediacloud_articles'
    collection = 'articles'
    doc = sphinx_query(index, query)
    doc_ids = [ObjectId(d["_id"]) for d in doc]
    documents = fetch_docs(doc_ids, collection)
