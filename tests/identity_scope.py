import gc
import weakref

from peewee import *

from .base import ModelTestCase
from .base import TestModel


class IdentityItem(TestModel):
    name = CharField()
    status = CharField(default='new')


class IdentityNoPK(TestModel):
    name = CharField()

    class Meta:
        primary_key = False


class IdentityCPK(TestModel):
    left = IntegerField()
    right = IntegerField()
    name = CharField()

    class Meta:
        primary_key = CompositeKey('left', 'right')


class IdentityParent(TestModel):
    name = CharField()


class IdentityChild(TestModel):
    parent = ForeignKeyField(IdentityParent, backref='children')
    name = CharField()


class IdentityScopeTestCase(ModelTestCase):
    requires = [
        IdentityItem,
        IdentityNoPK,
        IdentityCPK,
        IdentityParent,
        IdentityChild,
    ]

    def test_reuses_entities_and_fills_partial_columns(self):
        item = IdentityItem.create(name='first', status='old')

        with identity_scope() as scope:
            first = IdentityItem.get_by_id(item.id)
            second = IdentityItem.get_by_id(item.id)
            self.assertIs(first, second)

            partial = (IdentityItem
                       .select(IdentityItem.id, IdentityItem.status)
                       .where(IdentityItem.id == item.id)
                       .get())
            self.assertIs(partial, first)
            self.assertEqual(partial.id, item.id)
            self.assertEqual(partial.name, 'first')
            self.assertEqual(partial.status, 'old')
            self.assertIn((IdentityItem, (item.id,)), scope.identity_map)

        self.assertEqual(scope.identity_map, {})
        outside = IdentityItem.get_by_id(item.id)
        self.assertIsNot(outside, first)

    def test_database_values_fill_clean_fields_and_dirty_values_win(self):
        item = IdentityItem.create(name='first', status='old')

        with identity_scope() as scope:
            cached = IdentityItem.get_by_id(item.id)
            cached.status = 'local'

            IdentityItem.update(name='database').execute()
            refreshed = IdentityItem.get_by_id(item.id)

            self.assertIs(refreshed, cached)
            self.assertEqual(refreshed.name, 'database')
            self.assertEqual(refreshed.status, 'local')
            self.assertEqual(refreshed.dirty_field_names, [IdentityItem.status])
            self.assertEqual(scope.conflicts, [{
                'model': IdentityItem,
                'identity': (item.id,),
                'field': 'status',
                'database_value': 'old',
                'local_value': 'local',
            }])

    def test_projection_without_primary_key_is_not_merged(self):
        item = IdentityItem.create(name='first', status='old')

        with identity_scope() as scope:
            cached = IdentityItem.get_by_id(item.id)
            projection = (IdentityItem
                           .select(IdentityItem.name, IdentityItem.status)
                           .where(IdentityItem.id == item.id)
                           .get())

            self.assertIsNot(projection, cached)
            self.assertEqual(projection.__data__.keys(), {'name', 'status'})
            self.assertEqual(scope.identity_map, {
                (IdentityItem, (item.id,)): cached,
            })

    def test_model_without_primary_key_is_not_merged(self):
        IdentityNoPK.create(name='first')

        with identity_scope() as scope:
            first = IdentityNoPK.get()
            second = IdentityNoPK.get()

            self.assertIsNot(first, second)
            self.assertEqual(scope.identity_map, {})

    def test_composite_primary_key_identity(self):
        cpk = IdentityCPK.create(left=1, right=2, name='first')

        with identity_scope() as scope:
            first = IdentityCPK.get()
            second = IdentityCPK.get()
            self.assertIs(first, second)
            self.assertEqual(scope.identity_map, {
                (IdentityCPK, (1, 2)): first,
            })

            first.name = 'local'
            IdentityCPK.update(name='database').execute()
            refreshed = IdentityCPK.get()
            self.assertIs(refreshed, first)
            self.assertEqual(refreshed.name, 'local')
            self.assertEqual(scope.conflicts[0]['field'], 'name')

        self.assertEqual(cpk.left, 1)
        self.assertEqual(scope.identity_map, {})

    def test_join_reuses_entities_and_preserves_dirty_foreign_key(self):
        parent = IdentityParent.create(name='parent')
        child = IdentityChild.create(parent=parent, name='child')

        with identity_scope() as scope:
            cached_parent = IdentityParent.get_by_id(parent.id)
            cached_child = IdentityChild.get_by_id(child.id)
            clean_joined = (IdentityChild
                            .select(IdentityChild, IdentityParent)
                            .join(IdentityParent)
                            .where(IdentityChild.id == child.id)
                            .get())
            self.assertIs(clean_joined, cached_child)
            self.assertIs(clean_joined.parent, cached_parent)

            other_parent = IdentityParent.create(name='other')
            cached_child.parent = other_parent

            joined = (IdentityChild
                      .select(IdentityChild, IdentityParent)
                      .join(IdentityParent)
                      .where(IdentityChild.id == child.id)
                      .get())

            self.assertIs(joined, cached_child)
            self.assertEqual(joined.parent_id, other_parent.id)
            self.assertEqual(joined.parent.id, other_parent.id)
            self.assertEqual(joined.dirty_field_names, [IdentityChild.parent])
            self.assertEqual(scope.conflicts[0]['field'], 'parent')

    def test_model_alias_uses_base_model_identity(self):
        item = IdentityItem.create(name='first', status='old')
        item_alias = IdentityItem.alias()

        with identity_scope() as scope:
            cached = IdentityItem.get_by_id(item.id)
            aliased = (item_alias
                       .select(item_alias)
                       .where(item_alias.id == item.id)
                       .get())

            self.assertIs(aliased, cached)
            self.assertEqual(scope.identity_map, {
                (IdentityItem, (item.id,)): cached,
            })

    def test_nested_scopes_have_independent_ownership(self):
        item = IdentityItem.create(name='first')

        query = IdentityItem.select().where(IdentityItem.id == item.id)
        with identity_scope() as outer:
            outer_item = query.get()

            with identity_scope() as inner:
                inner_item = query.get()
                self.assertIsNot(inner_item, outer_item)
                self.assertEqual(inner.identity_map, {
                    (IdentityItem, (item.id,)): inner_item,
                })

            self.assertEqual(inner.identity_map, {})
            self.assertIs(query.get(), outer_item)

        self.assertIsNot(query.get(), outer_item)

    def test_scope_releases_identity_cache_on_exit(self):
        item = IdentityItem.create(name='first')

        with identity_scope() as scope:
            cached = IdentityItem.get_by_id(item.id)
            ref = weakref.ref(cached)

        self.assertEqual(scope.identity_map, {})
        del cached
        gc.collect()
        self.assertIsNone(ref())

    def test_plain_queries_remain_uncached(self):
        item = IdentityItem.create(name='first')
        first = IdentityItem.get_by_id(item.id)
        second = IdentityItem.get_by_id(item.id)

        self.assertIsNot(first, second)

    def test_dirty_primary_key_is_not_merged(self):
        item = IdentityItem.create(name='first')

        with identity_scope() as scope:
            cached = IdentityItem.get_by_id(item.id)
            cached.id = item.id + 100
            refreshed = IdentityItem.get_by_id(item.id)

            self.assertIsNot(refreshed, cached)
            self.assertEqual(cached.id, item.id + 100)
            self.assertEqual(refreshed.id, item.id)
            self.assertEqual(scope.identity_map, {
                (IdentityItem, (item.id,)): cached,
            })
            self.assertEqual(scope.identity_map[(IdentityItem, (item.id,))].id,
                             item.id + 100)

    def test_non_model_rows_are_not_cached(self):
        IdentityItem.create(name='first', status='old')

        with identity_scope():
            dict_row = IdentityItem.select().dicts().get()
            tuple_row = IdentityItem.select().tuples().get()

        self.assertIsInstance(dict_row, dict)
        self.assertIsInstance(tuple_row, tuple)
