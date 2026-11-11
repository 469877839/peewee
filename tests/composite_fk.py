"""
Composite foreign-key tests.

A composite foreign-key is a single relation spanning several columns that
points at a composite candidate key (a composite primary key or a UNIQUE
index) on the related model. The motivating example is an ingestion library
whose rows are identified by (source_id, external_id): two sources may carry
the same external_id, so relating on either column alone would silently mix
rows belonging to different sources.
"""
from peewee import *

from .base import ModelTestCase
from .base import requires_sqlite
from .base import TestModel


class Source(TestModel):
    name = CharField(unique=True)


class Record(TestModel):
    source = ForeignKeyField(Source, backref='records')
    external_id = CharField()
    payload = CharField(null=True)

    class Meta:
        # (source, external_id) is the candidate key the relation targets.
        indexes = ((('source', 'external_id'), True),)


class Item(TestModel):
    source = ForeignKeyField(Source, backref='items')
    external_id = CharField()
    note = CharField(null=True)
    record = CompositeForeignKeyField(
        Record, (('source', 'source'), ('external_id', 'external_id')),
        backref='items')


class NullableItem(TestModel):
    source = ForeignKeyField(Source, null=True, backref='nullable_items')
    external_id = CharField(null=True)
    note = CharField(null=True)
    record = CompositeForeignKeyField(
        Record, (('source', 'source'), ('external_id', 'external_id')),
        null=True)


class ReversedItem(TestModel):
    # Declared order is explicit: the DDL must follow it, not alphabetical or
    # primary-key order.
    source = ForeignKeyField(Source, backref='reversed_items')
    external_id = CharField()
    note = CharField(null=True)
    record = CompositeForeignKeyField(
        Record, (('external_id', 'external_id'), ('source', 'source')))


class CompositeRecord(TestModel):
    source = ForeignKeyField(Source, backref='+')
    external_id = CharField()
    payload = CharField(null=True)

    class Meta:
        primary_key = CompositeKey('source', 'external_id')


class CompositeKeyItem(TestModel):
    source = ForeignKeyField(Source, backref='composite_key_items')
    external_id = CharField()
    note = CharField(null=True)
    record = CompositeForeignKeyField(
        CompositeRecord, (('source', 'source'), ('external_id', 'external_id')),
        backref='items')


ALL_MODELS = [
    Source, Record, Item, NullableItem, ReversedItem, CompositeRecord,
    CompositeKeyItem,
]


class CompositeFKTestCase(ModelTestCase):
    requires = ALL_MODELS

    def setUp(self):
        super(CompositeFKTestCase, self).setUp()
        self.s1 = Source.create(name='source-1')
        self.s2 = Source.create(name='source-2')
        # The two sources share an external_id on purpose.
        self.r1 = Record.create(source=self.s1, external_id='ext-1',
                                payload='payload-1')
        self.r2 = Record.create(source=self.s2, external_id='ext-1',
                                payload='payload-2')
        self.r3 = Record.create(source=self.s1, external_id='ext-2',
                                payload='payload-3')
        self.i1 = Item.create(source=self.s1, external_id='ext-1',
                              note='item-1')
        self.i2 = Item.create(source=self.s2, external_id='ext-1',
                              note='item-2')
        self.i3 = Item.create(source=self.s1, external_id='ext-2',
                              note='item-3')


class TestCompositeForeignKeySchema(CompositeFKTestCase):
    @requires_sqlite
    def test_table_constraint(self):
        sql = self.database.execute_sql(
            'SELECT sql FROM sqlite_master WHERE name = ?',
            (Item._meta.table_name,)).fetchone()[0]
        self.assertIn(
            'FOREIGN KEY ("source_id", "external_id") '
            'REFERENCES "record" ("source_id", "external_id")', sql)

    @requires_sqlite
    def test_declared_pair_order_is_preserved(self):
        sql = self.database.execute_sql(
            'SELECT sql FROM sqlite_master WHERE name = ?',
            (ReversedItem._meta.table_name,)).fetchone()[0]
        self.assertIn(
            'FOREIGN KEY ("external_id", "source_id") '
            'REFERENCES "record" ("external_id", "source_id")', sql)

    @requires_sqlite
    def test_constraint_is_enforced(self):
        self.database.foreign_keys = True
        try:
            bad = Item(source=self.s1, external_id='missing')
            bad.record = (self.s1.id, 'missing')
            with self.assertRaises(IntegrityError):
                bad.save()
        finally:
            self.database.foreign_keys = False

    def test_composite_primary_key_is_a_candidate_key(self):
        rec = CompositeRecord.create(source=self.s1, external_id='pk-1',
                                     payload='pk-payload')
        item = CompositeKeyItem(source=self.s1, external_id='pk-1')
        item.record = rec
        item.save()
        self.assertEqual(item.record.payload, 'pk-payload')

    def test_non_unique_target_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not a composite primary '
                                   'key or a UNIQUE key'):
            class BadItem(TestModel):
                source = ForeignKeyField(Source)
                external_id = CharField()
                record = CompositeForeignKeyField(
                    Record, (('source', 'source'),
                             ('external_id', 'payload')))
                class Meta:
                    database = self.database

    def test_single_pair_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least two'):
            CompositeForeignKeyField(Record, (('source', 'source'),))

    def test_malformed_pairs_rejected(self):
        with self.assertRaisesRegex(ValueError, 'column pairs'):
            CompositeForeignKeyField(Record, (('source',),))
        with self.assertRaisesRegex(ValueError, 'column pairs'):
            CompositeForeignKeyField(
                Record, (('source', 'source', 'extra'),))

    def test_duplicate_local_column_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate local columns'):
            CompositeForeignKeyField(
                Record, (('source', 'source'), ('source', 'external_id')))

    def test_target_columns_must_exist(self):
        with self.assertRaisesRegex(ValueError, 'not defined on'):
            class BadItem(TestModel):
                source = ForeignKeyField(Source)
                external_id = CharField()
                record = CompositeForeignKeyField(
                    Record, (('source', 'source'),
                             ('external_id', 'nope')))
                class Meta:
                    database = self.database

    def test_two_composite_relations_cannot_share_a_column(self):
        with self.assertRaisesRegex(ValueError, 'cannot reuse column'):
            class Shared(TestModel):
                source = ForeignKeyField(Source)
                external_id = CharField()
                first = CompositeForeignKeyField(
                    Record, (('source', 'source'),
                             ('external_id', 'external_id')))
                second = CompositeForeignKeyField(
                    Record, (('source', 'source'),
                             ('external_id', 'external_id')))
                class Meta:
                    database = self.database

    def test_composite_field_is_not_a_column(self):
        self.assertNotIn('record', Item._meta.fields)
        self.assertNotIn('record', Item._meta.sorted_field_names)
        self.assertIn('source_id', Item._meta.columns)


class TestCompositeForeignKeyAccess(CompositeFKTestCase):
    def test_lazy_load_does_not_cross_sources(self):
        item1 = Item.get_by_id(self.i1.id)
        item2 = Item.get_by_id(self.i2.id)
        self.assertEqual(item1.record.payload, 'payload-1')
        self.assertEqual(item2.record.payload, 'payload-2')

    def test_lazy_load_is_cached(self):
        item = Item.get_by_id(self.i1.id)
        with self.assertQueryCount(1):
            first = item.record
            second = item.record
        self.assertIs(first, second)

    def test_assign_entity(self):
        item = Item(source=self.s2, external_id='ext-1', note='assigned')
        item.record = self.r2
        item.save()
        self.assertEqual(item.record.payload, 'payload-2')
        fetched = Item.get_by_id(item.id)
        self.assertEqual(fetched.record.payload, 'payload-2')

    def test_assign_tuple(self):
        item = Item(note='from-tuple')
        item.record = (self.s2.id, 'ext-1')
        item.save()
        self.assertEqual(item.record.payload, 'payload-2')
        self.assertEqual((item.source_id, item.external_id),
                         (self.s2.id, 'ext-1'))

    def test_assign_marks_columns_dirty(self):
        item = Item(source=self.s1, external_id='ext-1')
        item.record = (self.s2.id, 'ext-1')
        self.assertEqual(item.dirty_field_names, ['source', 'external_id'])

    def test_backref_queries_are_isolated(self):
        self.assertEqual([i.note for i in self.r1.items], ['item-1'])
        self.assertEqual([i.note for i in self.r2.items], ['item-2'])

    def test_join_isolated_by_both_columns(self):
        query = Item.select(Item, Record).join(Record, on=Item.record)
        rows = [(item.record.payload, item.note) for item in query]
        self.assertEqual(rows, [
            ('payload-1', 'item-1'),
            ('payload-2', 'item-2'),
            ('payload-3', 'item-3')])

    def test_join_infers_composite_relation(self):
        query = Item.select(Item, Record).join(Record)
        self.assertEqual([item.record.payload for item in query],
                         ['payload-1', 'payload-2', 'payload-3'])

    def test_backref_join(self):
        query = Record.select(Record, Item).join(Item)
        rows = sorted((record.payload, record.item.note) for record in query)
        self.assertEqual(rows, [
            ('payload-1', 'item-1'), ('payload-2', 'item-2'),
            ('payload-3', 'item-3')])

    def test_left_outer_join(self):
        Record.create(source=self.s1, external_id='ext-9', payload='payload-9')
        query = Record.select(Record, Item).join(Item, JOIN.LEFT_OUTER)
        unmatched = [r for r in query
                     if not hasattr(r, 'item') or r.item is None]
        self.assertEqual([r.payload for r in unmatched], ['payload-9'])

    def test_alias_join(self):
        ia = Item.alias('ia')
        ra = Record.alias('ra')
        rows = sorted(i.note for i in ia.select(ia, ra).join(ra, on=ia.record))
        self.assertEqual(rows, ['item-1', 'item-2', 'item-3'])

    def test_alias_comparison(self):
        ia = Item.alias('ia')
        query = ia.select(ia).where(ia.record == (self.s2.id, 'ext-1'))
        self.assertEqual([i.note for i in query], ['item-2'])

    def test_row_value_in(self):
        query = Item.select().where(Item.record << [
            (self.s1.id, 'ext-1'), (self.s2.id, 'ext-1')])
        self.assertEqual(sorted(i.note for i in query), ['item-1', 'item-2'])

    def test_row_value_in_subquery(self):
        sub = (Record
               .select(Record.source, Record.external_id)
               .where(Record.payload == 'payload-2'))
        query = Item.select().where(Item.record << sub)
        self.assertEqual([i.note for i in query], ['item-2'])

    def test_in_subquery_column_count_mismatch(self):
        sub = Record.select(Record.id)
        with self.assertRaisesRegex(ValueError, 'must select 2 columns'):
            Item.select().where(Item.record << sub)

    def test_in_partial_row_rejected(self):
        with self.assertRaisesRegex(ValueError, 'must have 2 columns'):
            Item.select().where(Item.record << [(self.s1.id,)])

    def test_compare_requires_a_tuple(self):
        with self.assertRaisesRegex(ValueError, 'tuple of 2 values'):
            Item.select().where(Item.record == 'ext-1')

    def test_composite_key_not_usable_in_insert(self):
        with self.assertRaisesRegex(ValueError, 'individual columns'):
            Item.insert({Item.record: self.r1})

    def test_composite_key_not_usable_in_update(self):
        with self.assertRaisesRegex(ValueError, 'individual columns'):
            Item.update({Item.record: self.r1})


class TestCompositeForeignKeyNulls(CompositeFKTestCase):
    def test_all_null_with_nullable_relation(self):
        item = NullableItem(note='null-rel')
        item.record = (None, None)
        item.save()
        self.assertIsNone(item.record)
        item2 = NullableItem(note='null-enum')
        item2.record = None
        item2.save()
        self.assertIsNone(item2.record)
        loaded = NullableItem.get_by_id(item.id)
        self.assertIsNone(loaded.record)

    def test_partial_null_rejected(self):
        item = NullableItem(note='bad')
        with self.assertRaisesRegex(ValueError, 'partially-NULL'):
            item.record = (self.s1.id, None)

    def test_not_null_relation_rejects_all_null(self):
        item = Item(note='bad')
        with self.assertRaisesRegex(ValueError, 'NOT NULL'):
            item.record = (None, None)

    def test_null_tuple_length_checked(self):
        item = NullableItem()
        with self.assertRaisesRegex(ValueError, 'expects 2 values'):
            item.record = (self.s1.id,)


class TestCompositeForeignKeyPrefetch(CompositeFKTestCase):
    def test_prefetch_forward(self):
        items = prefetch(Item.select(), Record)
        self.assertEqual([(i.note, i.record.payload) for i in items], [
            ('item-1', 'payload-1'), ('item-2', 'payload-2'),
            ('item-3', 'payload-3')])

    def test_prefetch_forward_join_strategy(self):
        items = prefetch(Item.select(), Record, prefetch_type=PREFETCH_TYPE.JOIN)
        self.assertEqual([(i.note, i.record.payload) for i in items], [
            ('item-1', 'payload-1'), ('item-2', 'payload-2'),
            ('item-3', 'payload-3')])

    def test_prefetch_backref(self):
        records = prefetch(Record.select(), Item)
        self.assertEqual(
            [(r.payload, [i.note for i in r.items]) for r in records],
            [('payload-1', ['item-1']), ('payload-2', ['item-2']),
             ('payload-3', ['item-3'])])

    def test_prefetch_backref_join_strategy(self):
        records = prefetch(Record.select(), Item,
                           prefetch_type=PREFETCH_TYPE.JOIN)
        self.assertEqual(
            [(r.payload, [i.note for i in r.items]) for r in records],
            [('payload-1', ['item-1']), ('payload-2', ['item-2']),
             ('payload-3', ['item-3'])])

    def test_prefetch_unmatched_parent(self):
        Record.create(source=self.s1, external_id='ext-9', payload='payload-9')
        records = prefetch(Record.select(), Item)
        self.assertEqual(
            [(r.payload, [i.note for i in r.items]) for r in records],
            [('payload-1', ['item-1']), ('payload-2', ['item-2']),
             ('payload-3', ['item-3']), ('payload-9', [])])

    def test_prefetch_query_count(self):
        with self.assertQueryCount(2):
            items = list(prefetch(Item.select(), Record))
        self.assertEqual(len(items), 3)

    def test_with_related_strategies(self):
        for strategy in (PREFETCH_TYPE.WHERE, PREFETCH_TYPE.JOIN,
                         PREFETCH_TYPE.MATERIALIZE):
            items = list(Item.select().with_related(
                Load(Item.record, strategy=strategy)))
            self.assertEqual([(i.note, i.record.payload) for i in items], [
                ('item-1', 'payload-1'), ('item-2', 'payload-2'),
                ('item-3', 'payload-3')])

    def test_with_related_backref_strategies(self):
        for strategy in (PREFETCH_TYPE.WHERE, PREFETCH_TYPE.JOIN,
                         PREFETCH_TYPE.MATERIALIZE):
            records = list(Record.select().with_related(
                Load(Record.items, strategy=strategy)))
            self.assertEqual(
                [(r.payload, [i.note for i in r.items]) for r in records],
                [('payload-1', ['item-1']), ('payload-2', ['item-2']),
                 ('payload-3', ['item-3'])])

    def test_per_parent_rejected(self):
        with self.assertRaisesRegex(ValueError, 'per_parent'):
            Load(Record.items, per_parent=1)

    def test_backref_cannot_be_join_condition(self):
        with self.assertRaisesRegex(ValueError, 'back-reference'):
            Record.select(Record, Item).join(Item, on=Record.items)


class TestCompositeForeignKeyDelete(CompositeFKTestCase):
    def test_delete_record_does_not_affect_other_source(self):
        # Both sources carry external_id='ext-1'. Deleting source-1's record
        # must not make source-2's item resolve to anything different.
        self.r1.delete_instance()
        item1 = Item.get_by_id(self.i1.id)
        with self.assertRaises(Record.DoesNotExist):
            item1.record
        item2 = Item.get_by_id(self.i2.id)
        self.assertEqual(item2.record.payload, 'payload-2')
        self.assertEqual(
            [r.payload for r in Record.select().order_by(Record.id)],
            ['payload-2', 'payload-3'])

    def test_delete_all_rows_of_one_source_is_isolated(self):
        Record.delete().where(Record.source == self.s1.id).execute()
        Item.delete().where(Item.source == self.s1.id).execute()
        # Source-2 is untouched even though external_id='ext-1' is shared.
        item2 = Item.get_by_id(self.i2.id)
        self.assertEqual((item2.source_id, item2.external_id),
                         (self.s2.id, 'ext-1'))
        self.assertEqual(item2.record.payload, 'payload-2')
        self.assertEqual([r.payload for r in Record.select()], ['payload-2'])
        self.assertEqual([i.note for i in Item.select()], ['item-2'])

    @requires_sqlite
    def test_referenced_record_cannot_be_deleted_with_fk_on(self):
        self.database.foreign_keys = True
        try:
            with self.assertRaises(IntegrityError):
                self.r1.delete_instance()
            item2 = Item.get_by_id(self.i2.id)
            self.assertEqual(item2.record.payload, 'payload-2')
        finally:
            self.database.foreign_keys = False
