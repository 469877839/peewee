"""Tests for the explicit IdentityMap entity-reuse scope.

Every test runs against an in-memory SQLite database. The scope is
thread-local and owns its cache, so the tests also assert that nothing the
scope materialized survives the ``with`` block.
"""
import gc
import threading
import weakref

from peewee import *
from peewee import IdentityMap

from .base import get_in_memory_db
from .base import ModelTestCase
from .base import TestModel


# ---------------------------------------------------------------------------
# Module-local models.
# ---------------------------------------------------------------------------

class IdentityUser(TestModel):
    username = CharField()
    data = CharField(null=True)


class IdentityTweet(TestModel):
    user = ForeignKeyField(IdentityUser, backref='tweets', null=True)
    content = TextField()
    extra = CharField(null=True)


class IdentityPair(TestModel):
    left = CharField()
    right = IntegerField()
    payload = CharField(null=True)

    class Meta:
        primary_key = CompositeKey('left', 'right')


class IdentityTotal(TestModel):
    # No primary key: rows are mere projections and must not be merged.
    bucket = CharField()
    amount = IntegerField()

    class Meta:
        primary_key = False


class IdentityEmployee(TestModel):
    name = CharField()
    manager = ForeignKeyField('self', null=True, backref='reports')


MODELS = [IdentityUser, IdentityTweet, IdentityPair, IdentityTotal,
          IdentityEmployee]


class IdentityMapTestCase(ModelTestCase):
    database = get_in_memory_db()
    requires = MODELS

    def setUp(self):
        super(IdentityMapTestCase, self).setUp()
        with self.database.atomic():
            self.huey = IdentityUser.create(username='huey', data='d1')
            self.zaizee = IdentityUser.create(username='zaizee', data='d2')
            self.tweet1 = IdentityTweet.create(
                user=self.huey, content='meow', extra='x1')
            self.tweet2 = IdentityTweet.create(
                user=self.huey, content='purr', extra='x2')
            self.tweet3 = IdentityTweet.create(
                user=self.zaizee, content='hiss', extra='x3')
            IdentityPair.create(left='a', right=1, payload='p1')
            IdentityPair.create(left='a', right=2, payload='p2')
            IdentityTotal.create(bucket='b', amount=1)
            IdentityTotal.create(bucket='b', amount=1)
            self.boss = IdentityEmployee.create(name='boss')
            self.worker = IdentityEmployee.create(
                name='worker', manager=self.boss)


class TestIdentityMapReuse(IdentityMapTestCase):
    def test_is_relation_across_queries(self):
        with IdentityMap() as scope:
            a = IdentityUser.get_by_id(self.huey.id)
            b = IdentityUser.get(IdentityUser.username == 'huey')
            self.assertIs(a, b)
            self.assertEqual(a.username, 'huey')
            self.assertEqual(a.data, 'd1')
            self.assertEqual(len(scope), 1)

    def test_list_query_shares_identity_with_get(self):
        with IdentityMap():
            users = list(IdentityUser.select().order_by(IdentityUser.id))
            by_id = IdentityUser.get_by_id(self.huey.id)
            self.assertIs(users[0], by_id)
            self.assertEqual(len(users), 2)

    def test_partial_projection_fills_remaining_fields(self):
        with IdentityMap():
            # Only the primary key is selected - the row cannot hydrate the
            # other fields on its own.
            thin = (IdentityUser
                    .select(IdentityUser.id)
                    .where(IdentityUser.id == self.huey.id)
                    .get())
            self.assertEqual(thin.username, None)
            # A later full row populates the same instance in place.
            full = IdentityUser.get_by_id(self.huey.id)
            self.assertIs(full, thin)
            self.assertEqual(full.username, 'huey')
            self.assertEqual(full.data, 'd1')

    def test_partial_projection_does_not_erase_fields(self):
        with IdentityMap():
            full = IdentityUser.get_by_id(self.huey.id)
            part = (IdentityUser
                    .select(IdentityUser.id, IdentityUser.data)
                    .where(IdentityUser.id == self.huey.id)
                    .get())
            self.assertIs(part, full)
            self.assertEqual(part.data, 'd1')
            # 'username' was not selected, so the cached value is untouched.
            self.assertEqual(part.username, 'huey')

    def test_database_values_refresh_only_selected_fields(self):
        with IdentityMap():
            user = IdentityUser.get_by_id(self.huey.id)
            (IdentityUser
             .update(username='huey2', data='d2')
             .where(IdentityUser.id == self.huey.id)
             .execute())
            part = (IdentityUser
                    .select(IdentityUser.id, IdentityUser.username)
                    .where(IdentityUser.id == self.huey.id)
                    .get())
            self.assertIs(part, user)
            self.assertEqual(part.username, 'huey2')   # re-read from db.
            self.assertEqual(part.data, 'd1')           # not selected.
            self.assertFalse(part.is_dirty())

            full = IdentityUser.get_by_id(self.huey.id)
            self.assertIs(full, user)
            self.assertEqual(full.data, 'd2')           # now refreshed.

    def test_join_query_reuses_root_and_related_identities(self):
        with IdentityMap():
            user = IdentityUser.get_by_id(self.huey.id)
            joined = (IdentityTweet
                      .select(IdentityTweet, IdentityUser)
                      .join(IdentityUser)
                      .where(IdentityTweet.id == self.tweet1.id)
                      .get())
            self.assertIs(joined.user, user)
            self.assertEqual(joined.content, 'meow')
            # The joined tweet is itself reusable across later queries.
            direct = IdentityTweet.get_by_id(self.tweet1.id)
            self.assertIs(direct, joined)
            # And so is the user it pulled in.
            self.assertIs(IdentityUser.get_by_id(self.huey.id), user)

    def test_alias_shares_underlying_identity(self):
        Manager = IdentityEmployee.alias()
        with IdentityMap():
            employee = IdentityEmployee.get_by_id(self.worker.id)
            joined = (IdentityEmployee
                      .select(IdentityEmployee, Manager)
                      .join(Manager, JOIN.LEFT_OUTER,
                            on=(IdentityEmployee.manager == Manager.id)
                            .alias('manager'))
                      .where(IdentityEmployee.id == self.worker.id)
                      .get())
            self.assertIs(joined, employee)
            manager = IdentityEmployee.get_by_id(self.boss.id)
            self.assertIs(joined.manager, manager)
            # Same row seen through the alias and the model is one object.
            aliased = (IdentityEmployee
                       .select(Manager)
                       .join(Manager, JOIN.LEFT_OUTER,
                             on=(IdentityEmployee.manager == Manager.id)
                             .alias('manager'))
                       .where(IdentityEmployee.id == self.worker.id)
                       .get())
            self.assertIs(aliased.manager, manager)

    def test_outer_join_null_side_is_not_cached(self):
        lonely = IdentityTweet.create(user=None, content='lone', extra='x0')
        with IdentityMap() as scope:
            rows = list(IdentityTweet
                        .select(IdentityTweet, IdentityUser)
                        .join(IdentityUser, JOIN.LEFT_OUTER)
                        .order_by(IdentityTweet.id))
            # 4 tweets + 2 joined users: the null user side never produces
            # a cached (IdentityUser, None) entity.
            self.assertEqual(len(scope), 6)
            lonely_row = [r for r in rows if r.id == lonely.id][0]
            self.assertIsNone(lonely_row.user)
            self.assertIsNone(lonely_row.__rel__.get('user'))
            self.assertNotIn((IdentityUser, None), scope._entities)
            # The orphan tweet itself is still keyed and reused.
            again = list(IdentityTweet
                         .select(IdentityTweet, IdentityUser)
                         .join(IdentityUser, JOIN.LEFT_OUTER)
                         .order_by(IdentityTweet.id))
            self.assertIs([r for r in again if r.id == lonely.id][0],
                          lonely_row)

    def test_composite_primary_key_reuse(self):
        with IdentityMap():
            first = IdentityPair.get(IdentityPair.left == 'a',
                                     IdentityPair.right == 1)
            same = IdentityPair.get_by_id(('a', 1))
            other = IdentityPair.get_by_id(('a', 2))
            self.assertIs(first, same)
            self.assertIsNot(first, other)
            # Partial composite projection (only one PK column) is not merged.
            x = (IdentityPair
                 .select(IdentityPair.left)
                 .where(IdentityPair.left == 'a')
                 .get())
            y = (IdentityPair
                 .select(IdentityPair.left)
                 .where(IdentityPair.left == 'a')
                 .get())
            self.assertIsNot(x, y)
            # Full composite key present, payload retained from first load.
            part = (IdentityPair
                    .select(IdentityPair.left, IdentityPair.right)
                    .where((IdentityPair.left == 'a') &
                           (IdentityPair.right == 1))
                    .get())
            self.assertIs(part, first)
            self.assertEqual(part.payload, 'p1')

    def test_composite_key_dirty_parts_win(self):
        with IdentityMap():
            pair = IdentityPair.get_by_id(('a', 1))
            pair.payload = 'local'
            again = IdentityPair.get_by_id(('a', 1))
            self.assertIs(again, pair)
            self.assertEqual(again.payload, 'local')
            self.assertEqual(again.dirty_field_names, ['payload'])


class TestIdentityMapDirtyFields(IdentityMapTestCase):
    def test_local_changes_win_over_database_values(self):
        with IdentityMap():
            user = IdentityUser.get_by_id(self.huey.id)
            user.username = 'local-name'
            (IdentityUser
             .update(username='db-name', data='db-data')
             .where(IdentityUser.id == self.huey.id)
             .execute())
            again = IdentityUser.get_by_id(self.huey.id)
            self.assertIs(again, user)
            self.assertEqual(again.username, 'local-name')  # local wins.
            self.assertEqual(again.data, 'db-data')          # clean refresh.
            self.assertEqual(again.dirty_field_names, ['username'])

    def test_dirty_field_stays_dirty_even_when_values_agree(self):
        with IdentityMap():
            user = IdentityUser.get_by_id(self.huey.id)
            user.data = 'd1'  # Assigning marks the field dirty.
            self.assertEqual(user.dirty_field_names, ['data'])
            again = IdentityUser.get_by_id(self.huey.id)
            self.assertIs(again, user)
            self.assertEqual(again.dirty_field_names, ['data'])

    def test_dirty_primary_key_is_not_overwritten(self):
        with IdentityMap():
            user = IdentityUser.get_by_id(self.huey.id)
            user.id = 1000
            again = IdentityUser.get_by_id(self.huey.id)
            self.assertIs(again, user)
            self.assertEqual(again.id, 1000)  # Local edit preserved.
            self.assertEqual(again.dirty_field_names, ['id'])

    def test_dirty_foreign_key_survives_join(self):
        with IdentityMap():
            tweet = IdentityTweet.get_by_id(self.tweet1.id)
            tweet.user = self.zaizee
            joined = (IdentityTweet
                      .select(IdentityTweet, IdentityUser)
                      .join(IdentityUser)
                      .where(IdentityTweet.id == self.tweet1.id)
                      .get())
            self.assertIs(joined, tweet)
            self.assertIs(joined.user, self.zaizee)
            self.assertEqual(joined.user_id, self.zaizee.id)
            self.assertIn('user', joined._dirty)

    def test_dirty_raw_fk_id_survives_join(self):
        with IdentityMap():
            tweet = IdentityTweet.get_by_id(self.tweet1.id)
            tweet.user_id = self.zaizee.id
            self.assertIn('user', tweet._dirty)
            joined = (IdentityTweet
                      .select(IdentityTweet, IdentityUser)
                      .join(IdentityUser)
                      .where(IdentityTweet.id == self.tweet1.id)
                      .get())
            self.assertIs(joined, tweet)
            self.assertEqual(joined.user_id, self.zaizee.id)
            self.assertIn('user', joined._dirty)

    def test_clean_relation_is_refreshed_without_becoming_dirty(self):
        with IdentityMap():
            tweet = IdentityTweet.get_by_id(self.tweet1.id)
            self.assertFalse(tweet.is_dirty())
            joined = (IdentityTweet
                      .select(IdentityTweet, IdentityUser)
                      .join(IdentityUser)
                      .where(IdentityTweet.id == self.tweet1.id)
                      .get())
            self.assertIs(joined, tweet)
            self.assertIs(joined.user, IdentityUser.get_by_id(self.huey.id))
            self.assertFalse(joined.is_dirty())
            self.assertNotIn('user', joined._dirty)


class TestIdentityMapProjections(IdentityMapTestCase):
    def test_projection_without_primary_key_is_not_merged(self):
        def select_username():
            return list(IdentityUser
                        .select(IdentityUser.username)
                        .order_by(IdentityUser.username))

        def get_username():
            return (IdentityUser
                    .select(IdentityUser.username)
                    .where(IdentityUser.id == self.huey.id)
                    .get())

        with IdentityMap() as scope:
            # 'username' alone carries no primary key, so no identity key
            # can be derived and the rows stay standalone.
            rows = select_username()
            again = select_username()
            for a, b in zip(rows, again):
                self.assertIsNot(a, b)
            self.assertEqual(len(scope), 0)
            # The same projection via .get() yields two separate objects.
            first = get_username()
            second = get_username()
            self.assertIsNot(first, second)
            self.assertEqual(first.username, second.username)
            # Including the primary key makes the projection mergeable again.
            keyed = (IdentityUser
                     .select(IdentityUser.id, IdentityUser.username)
                     .where(IdentityUser.id == self.huey.id)
                     .get())
            self.assertIs(keyed,
                          IdentityUser.get_by_id(self.huey.id))

    def test_model_without_primary_key_is_not_merged(self):
        with IdentityMap() as scope:
            a = IdentityTotal.get(IdentityTotal.amount == 1)
            b = IdentityTotal.get(IdentityTotal.amount == 1)
            self.assertIsNot(a, b)
            self.assertEqual(len(scope), 0)

    def test_non_model_row_types_are_untouched(self):
        with IdentityMap() as scope:
            query = (IdentityUser
                     .select(IdentityUser.id, IdentityUser.username)
                     .where(IdentityUser.id == self.huey.id))
            d1 = query.dicts().get()
            d2 = query.dicts().get()
            t1 = query.tuples().get()
            t2 = query.tuples().get()
            self.assertIsNot(d1, d2)
            self.assertIsNot(t1, t2)
            # Only model instances take part in identity reuse.
            self.assertEqual(len(scope), 0)

    def test_column_alias_without_pk_is_not_merged(self):
        def aliased():
            return (IdentityUser
                    .select(IdentityUser.id.alias('uid'),
                            IdentityUser.username)
                    .where(IdentityUser.id == self.huey.id)
                    .get())

        with IdentityMap() as scope:
            # The primary-key column is renamed to an alias, so the row no
            # longer exposes the model's pk field and cannot be merged.
            a = aliased()
            b = aliased()
            self.assertIsNot(a, b)
            self.assertEqual(a.uid, self.huey.id)
            self.assertEqual(len(scope), 0)

            # Selecting the pk field itself (no alias) restores merging.
            keyed = (IdentityUser
                     .select(IdentityUser.id,
                             IdentityUser.username.alias('handle'))
                     .where(IdentityUser.id == self.huey.id)
                     .get())
            self.assertIs(keyed,
                          IdentityUser.get_by_id(self.huey.id))
            self.assertEqual(keyed.handle, 'huey')


class TestIdentityMapLifetime(IdentityMapTestCase):
    def test_scope_exit_restores_isolation(self):
        with IdentityMap() as scope:
            a = IdentityUser.get_by_id(self.huey.id)
            b = IdentityUser.get_by_id(self.huey.id)
            self.assertIs(a, b)
        c = IdentityUser.get_by_id(self.huey.id)
        d = IdentityUser.get_by_id(self.huey.id)
        self.assertIsNot(c, d)
        self.assertEqual(len(scope), 0)  # References dropped on exit.
        self.assertIsNone(IdentityMap.current())

    def test_scope_releases_instances(self):
        with IdentityMap():
            user = IdentityUser.get_by_id(self.huey.id)
            ref = weakref.ref(user)
            self.assertIsNotNone(ref())
            del user
        gc.collect()
        self.assertIsNone(ref())

    def test_nested_scopes_own_their_instances(self):
        with IdentityMap() as outer:
            a = IdentityUser.get_by_id(self.huey.id)
            with IdentityMap() as inner:
                self.assertIs(IdentityMap.current(), inner)
                b = IdentityUser.get_by_id(self.huey.id)
                self.assertIsNot(a, b)  # Inner scope has its own entity.
                self.assertEqual(len(inner), 1)
                inner_again = IdentityUser.get_by_id(self.huey.id)
                self.assertIs(inner_again, b)
            self.assertEqual(len(inner), 0)  # Inner released on exit.
            self.assertIs(IdentityMap.current(), outer)
            c = IdentityUser.get_by_id(self.huey.id)
            self.assertIs(c, a)  # Outer cache still intact.
            self.assertEqual(len(outer), 1)  # Inner never leaked into outer.

    def test_scopes_do_not_share_cache(self):
        # Two scopes must not share a backing mapping.
        self.assertIsNot(IdentityMap()._entities, IdentityMap()._entities)
        scopes = []
        objects = []
        for _ in range(3):
            scope = IdentityMap()
            with scope:
                obj = IdentityUser.get_by_id(self.huey.id)
                objects.append(obj)
                self.assertEqual(len(scope), 1)
            scopes.append(scope)
        self.assertIsNot(objects[0], objects[1])
        self.assertIsNot(objects[1], objects[2])
        # Every scope dropped its entries rather than retaining a global cache.
        for scope in scopes:
            self.assertEqual(len(scope), 0)
        self.assertFalse(any(isinstance(value, dict)
                             for value in vars(IdentityMap).values()))

    def test_no_residual_state_after_exit(self):
        with IdentityMap():
            IdentityUser.get_by_id(self.huey.id)
        self.assertIsNone(IdentityMap.current())
        self.assertFalse(hasattr(IdentityMap._state, 'stack'))
        self.assertIsNone(getattr(IdentityMap._state, 'stack', None))

    def test_scope_is_thread_local(self):
        seen = []

        def worker():
            seen.append(IdentityMap.current())

        with IdentityMap():
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            self.assertIs(IdentityMap.current(), IdentityMap.current())
        self.assertEqual(seen, [None])

    def test_decorator_form(self):
        @IdentityMap()
        def lookup():
            return (IdentityUser.get_by_id(self.huey.id),
                    IdentityUser.get_by_id(self.huey.id))

        first, second = lookup()
        self.assertIs(first, second)
        self.assertIsNone(IdentityMap.current())
        self.assertIsNot(IdentityUser.get_by_id(self.huey.id),
                         IdentityUser.get_by_id(self.huey.id))

    def test_streaming_query_reuses_entities(self):
        with IdentityMap():
            users = list(IdentityUser.select()
                         .order_by(IdentityUser.id)
                         .iterator())
            self.assertIs(users[0],
                          IdentityUser.get_by_id(self.huey.id))
            self.assertIs(users[1],
                          IdentityUser.get_by_id(self.zaizee.id))

    def test_compound_query_reuses_entities(self):
        with IdentityMap():
            user = IdentityUser.get_by_id(self.huey.id)
            union = (IdentityUser
                     .select()
                     .where(IdentityUser.id == self.huey.id)
                     .union(IdentityUser
                            .select()
                            .where(IdentityUser.id == self.zaizee.id)))
            rows = list(union)
            self.assertIs(rows[0], user)

    def test_queries_outside_scope_keep_existing_behavior(self):
        a = IdentityUser.get_by_id(self.huey.id)
        b = IdentityUser.get_by_id(self.huey.id)
        self.assertIsNot(a, b)
        a.username = 'local'
        self.assertEqual(a.dirty_field_names, ['username'])
        # No scope -> the database row is read fresh each time.
        (IdentityUser
         .update(username='db')
         .where(IdentityUser.id == self.huey.id)
         .execute())
        c = IdentityUser.get_by_id(self.huey.id)
        self.assertEqual(c.username, 'db')

    def test_identity_map_adds_no_queries(self):
        with IdentityMap():
            IdentityUser.get_by_id(self.huey.id)
            with self.assertQueryCount(1):
                IdentityUser.get_by_id(self.huey.id)
