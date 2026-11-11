"""
Prefetch / N+1 query optimization tests.

All models in this module except Package/PackageItem are local (not from
base_models), because prefetch tests need specific relational graph shapes
that differ from the shared models.
"""
import gc
import sqlite3
import weakref

from peewee import *

from .base import requires_models
from .base import ModelTestCase
from .base import TestModel
from .base_models import Package
from .base_models import PackageItem


# prefetch() supports the strategies that embed the parent as a subquery.
# MATERIALIZE is excluded - it reads keys from already-fetched parents, which
# only with_related() holds.
SUBQUERY_PREFETCH_TYPES = (PREFETCH_TYPE.WHERE, PREFETCH_TYPE.JOIN)


# ---------------------------------------------------------------------------
# Module-local models for core prefetch tests.
# ---------------------------------------------------------------------------

class Person(TestModel):
    name = TextField()


class Relationship(TestModel):
    from_person = ForeignKeyField(Person, backref='relationships')
    to_person = ForeignKeyField(Person, backref='related_to')


class Note(TestModel):
    person = ForeignKeyField(Person, backref='notes')
    content = TextField()


class NoteItem(TestModel):
    note = ForeignKeyField(Note, backref='items')
    content = TextField()


class Like(TestModel):
    person = ForeignKeyField(Person, backref='likes')
    note = ForeignKeyField(Note, backref='likes')


class Flag(TestModel):
    note = ForeignKeyField(Note, backref='flags')
    is_spam = BooleanField()


class Category(TestModel):
    name = TextField()
    parent = ForeignKeyField('self', backref='children', null=True)


# ===========================================================================
# Core prefetch tests
# ===========================================================================

class TestPrefetch(ModelTestCase):
    requires = [Person, Note, NoteItem, Like, Flag]

    def create_test_data(self):
        data = {
            'huey': (
                ('meow', ('meow-1', 'meow-2', 'meow-3')),
                ('purr', ()),
                ('hiss', ('hiss-1', 'hiss-2'))),
            'mickey': (
                ('woof', ()),
                ('bark', ('bark-1', 'bark-2'))),
            'zaizee': (),
        }
        for name, notes in sorted(data.items()):
            person = Person.create(name=name)
            for note, items in notes:
                note = Note.create(person=person, content=note)
                for item in items:
                    NoteItem.create(note=note, content=item)

        Flag.create(note=Note.get(Note.content == 'purr'), is_spam=True)
        Flag.create(note=Note.get(Note.content == 'woof'), is_spam=True)

        Like.create(note=Note.get(Note.content == 'meow'),
                    person=Person.get(Person.name == 'mickey'))
        Like.create(note=Note.get(Note.content == 'woof'),
                    person=Person.get(Person.name == 'huey'))

    def setUp(self):
        super(TestPrefetch, self).setUp()
        self.create_test_data()

    def accumulate_results(self, query, sort_items=False):
        accum = []
        for person in query:
            notes = []
            for note in person.notes:
                items = []
                for item in note.items:
                    items.append(item.content)
                if sort_items:
                    items.sort()
                notes.append((note.content, items))
            if sort_items:
                notes.sort()
            accum.append((person.name, notes))
        return accum

    def test_prefetch_simple(self):
        for pt in SUBQUERY_PREFETCH_TYPES:
            with self.assertQueryCount(3):
                people = Person.select().order_by(Person.name)
                query = people.prefetch(Note, NoteItem, prefetch_type=pt)
                accum = self.accumulate_results(query, sort_items=True)

            self.assertEqual(accum, [
                ('huey', [
                    ('hiss', ['hiss-1', 'hiss-2']),
                    ('meow', ['meow-1', 'meow-2', 'meow-3']),
                    ('purr', [])]),
                ('mickey', [
                    ('bark', ['bark-1', 'bark-2']),
                    ('woof', [])]),
                ('zaizee', []),
            ])

    def test_prefetch_filter(self):
        for pt in SUBQUERY_PREFETCH_TYPES:
            with self.assertQueryCount(3):
                people = Person.select().order_by(Person.name)
                notes = (Note
                         .select()
                         .where(Note.content.not_in(('hiss', 'meow', 'woof')))
                         .order_by(Note.content.desc()))
                items = NoteItem.select().where(
                    ~NoteItem.content.endswith('-2'))
                query = prefetch(people, notes, items, prefetch_type=pt)
                self.assertEqual(self.accumulate_results(query), [
                    ('huey', [('purr', [])]),
                    ('mickey', [('bark', ['bark-1'])]),
                    ('zaizee', []),
                ])

    def test_prefetch_reverse(self):
        for pt in SUBQUERY_PREFETCH_TYPES:
            with self.assertQueryCount(2):
                people = Person.select().order_by(Person.name)
                notes = Note.select().order_by(Note.content)
                query = prefetch(notes, people, prefetch_type=pt)
                accum = [(note.content, note.person.name) for note in query]
                self.assertEqual(accum, [
                    ('bark', 'mickey'),
                    ('hiss', 'huey'),
                    ('meow', 'huey'),
                    ('purr', 'huey'),
                    ('woof', 'mickey')])

    def test_prefetch_reverse_with_parent_join(self):
        for pt in SUBQUERY_PREFETCH_TYPES:
            with self.assertQueryCount(2):
                notes = (Note
                         .select(Note, Person)
                         .join(Person)
                         .order_by(Note.content))
                items = NoteItem.select().order_by(NoteItem.content.desc())
                query = prefetch(notes, items, prefetch_type=pt)
                accum = [(note.person.name,
                          note.content,
                          [item.content for item in note.items])
                         for note in query]
                self.assertEqual(accum, [
                    ('mickey', 'bark', ['bark-2', 'bark-1']),
                    ('huey', 'hiss', ['hiss-2', 'hiss-1']),
                    ('huey', 'meow', ['meow-3', 'meow-2', 'meow-1']),
                    ('huey', 'purr', []),
                    ('mickey', 'woof', []),
                ])

    def test_prefetch_multi_depth(self):
        for pt in SUBQUERY_PREFETCH_TYPES:
            people = Person.select().order_by(Person.name)
            notes = Note.select().order_by(Note.content)
            items = NoteItem.select().order_by(NoteItem.content)
            flags = Flag.select().order_by(Flag.id)

            LikePerson = Person.alias('lp')
            likes = (Like
                     .select(Like, LikePerson.name)
                     .join(LikePerson, on=(Like.person == LikePerson.id)))

            # Five queries:
            # - person (outermost query)
            # - notes for people
            # - items for notes
            # - flags for notes
            # - likes for notes (includes join to person)
            with self.assertQueryCount(5):
                query = prefetch(people, notes, items, flags, likes,
                                 prefetch_type=pt)
                accum = []
                for person in query:
                    notes = []
                    for note in person.notes:
                        items = [item.content for item in note.items]
                        likes = [like.person.name for like in note.likes]
                        flags = [flag.is_spam for flag in note.flags]
                        notes.append((note.content, items, likes, flags))
                    accum.append((person.name, notes))

            self.assertEqual(accum, [
                ('huey', [
                    ('hiss', ['hiss-1', 'hiss-2'], [], []),
                    ('meow', ['meow-1', 'meow-2', 'meow-3'], ['mickey'], []),
                    ('purr', [], [], [True])]),
                ('mickey', [
                    ('bark', ['bark-1', 'bark-2'], [], []),
                    ('woof', [], ['huey'], [True])]),
                (u'zaizee', []),
            ])

    def test_prefetch_multi_depth_no_join(self):
        for pt in SUBQUERY_PREFETCH_TYPES:
            LikePerson = Person.alias()
            people = Person.select().order_by(Person.name)
            notes = Note.select().order_by(Note.content)
            items = NoteItem.select().order_by(NoteItem.content)
            flags = Flag.select().order_by(Flag.id)

            with self.assertQueryCount(6):
                query = prefetch(people, notes, items, flags, Like, LikePerson,
                                 prefetch_type=pt)
                accum = []
                for person in query:
                    notes = []
                    for note in person.notes:
                        items = [item.content for item in note.items]
                        likes = [like.person.name for like in note.likes]
                        flags = [flag.is_spam for flag in note.flags]
                        notes.append((note.content, items, likes, flags))
                    accum.append((person.name, notes))

            self.assertEqual(accum, [
                ('huey', [
                    ('hiss', ['hiss-1', 'hiss-2'], [], []),
                    ('meow', ['meow-1', 'meow-2', 'meow-3'], ['mickey'], []),
                    ('purr', [], [], [True])]),
                ('mickey', [
                    ('bark', ['bark-1', 'bark-2'], [], []),
                    ('woof', [], ['huey'], [True])]),
                (u'zaizee', []),
            ])

    def test_prefetch_with_group_by(self):
        for pt in SUBQUERY_PREFETCH_TYPES:
            people = (Person
                      .select(Person, fn.COUNT(Note.id).alias('note_count'))
                      .join(Note, JOIN.LEFT_OUTER)
                      .group_by(Person)
                      .order_by(Person.name))
            notes = Note.select().order_by(Note.content)
            items = NoteItem.select().order_by(NoteItem.content)
            with self.assertQueryCount(3):
                query = prefetch(people, notes, items, prefetch_type=pt)
                self.assertEqual(self.accumulate_results(query), [
                    ('huey', [
                        ('hiss', ['hiss-1', 'hiss-2']),
                        ('meow', ['meow-1', 'meow-2', 'meow-3']),
                        ('purr', [])]),
                    ('mickey', [
                        ('bark', ['bark-1', 'bark-2']),
                        ('woof', [])]),
                    ('zaizee', []),
                ])

                huey, mickey, zaizee = query
                self.assertEqual(huey.note_count, 3)
                self.assertEqual(mickey.note_count, 2)
                self.assertEqual(zaizee.note_count, 0)

    @requires_models(Category)
    def test_prefetch_self_join(self):
        def cc(name, parent=None):
            return Category.create(name=name, parent=parent)
        root = cc('root')
        p1 = cc('p1', root)
        p2 = cc('p2', root)
        for p in (p1, p2):
            for i in range(2):
                cc('%s-%s' % (p.name, i + 1), p)

        for pt in SUBQUERY_PREFETCH_TYPES:
            Child = Category.alias('child')
            with self.assertQueryCount(2):
                query = prefetch(Category.select().order_by(Category.id),
                                 Child, prefetch_type=pt)
                names_and_children = [
                    (cat.name, [child.name for child in cat.children])
                    for cat in query]

            self.assertEqual(names_and_children, [
                ('root', ['p1', 'p2']),
                ('p1', ['p1-1', 'p1-2']),
                ('p2', ['p2-1', 'p2-2']),
                ('p1-1', []),
                ('p1-2', []),
                ('p2-1', []),
                ('p2-2', []),
            ])

    @requires_models(Category)
    def test_prefetch_adjacency_list(self):
        def cc(name, parent=None):
            return Category.create(name=name, parent=parent)

        tree = ('root', (
            ('n1', (
                ('c11', ()),
                ('c12', ()))),
            ('n2', (
                ('c21', ()),
                ('c22', (
                    ('g221', ()),
                    ('g222', ()))),
                ('c23', ()),
                ('c24', (
                    ('g241', ()),
                    ('g242', ()),
                    ('g243', ())))))))
        stack = [(None, tree)]
        while stack:
            parent, (name, children) = stack.pop()
            node = cc(name, parent)
            for child_tree in children:
                stack.insert(0, (node, child_tree))

        for pt in SUBQUERY_PREFETCH_TYPES:
            C = Category.alias('c')
            G = Category.alias('g')
            GG = Category.alias('gg')
            GGG = Category.alias('ggg')
            query = Category.select().where(Category.name == 'root')
            with self.assertQueryCount(5):
                pf = prefetch(query, C, (G, C), (GG, G), (GGG, GG),
                              prefetch_type=pt)
                def gather(c):
                    children = sorted([gather(ch) for ch in c.children])
                    return (c.name, tuple(children))
                nodes = list(pf)
                self.assertEqual(len(nodes), 1)
                pf_tree = gather(nodes[0])

            self.assertEqual(tree, pf_tree)

    def test_prefetch_specific_model(self):
        # Person -> Note
        #        -> Like (has fks to both person and note)
        Like.create(note=Note.get(Note.content == 'woof'),
                    person=Person.get(Person.name == 'zaizee'))
        NoteAlias = Note.alias('na')

        for pt in SUBQUERY_PREFETCH_TYPES:
            with self.assertQueryCount(3):
                people = Person.select().order_by(Person.name)
                notes = Note.select().order_by(Note.content)
                likes = (Like
                         .select(Like, NoteAlias.content)
                         .join(NoteAlias, on=(Like.note == NoteAlias.id))
                         .order_by(NoteAlias.content))
                query = prefetch(people, notes, (likes, Person),
                                 prefetch_type=pt)
                accum = []
                for person in query:
                    likes = []
                    notes = []
                    for note in person.notes:
                        notes.append(note.content)
                    for like in person.likes:
                        likes.append(like.note.content)
                    accum.append((person.name, notes, likes))

            self.assertEqual(accum, [
                ('huey', ['hiss', 'meow', 'purr'], ['woof']),
                ('mickey', ['bark', 'woof'], ['meow']),
                ('zaizee', [], ['woof']),
            ])

    @requires_models(Relationship)
    def test_multiple_foreign_keys(self):
        for model in (Like, Flag, NoteItem, Note, Person):
            model.delete().execute()
        c, h, z = [Person.create(name=name) for name in
                                 ('charlie', 'huey', 'zaizee')]
        RC = lambda f, t: Relationship.create(from_person=f, to_person=t)
        r1 = RC(c, h)
        r2 = RC(c, z)
        r3 = RC(h, c)
        r4 = RC(z, c)

        def assertRelationships(attr, values):
            self.assertEqual(len(attr),len(values))
            for relationship, value in zip(attr, values):
                self.assertEqual(relationship.__data__, value)

        for pt in SUBQUERY_PREFETCH_TYPES:
            with self.assertQueryCount(2):
                people = Person.select().order_by(Person.name)
                relationships = Relationship.select().order_by(Relationship.id)
                query = prefetch(people, relationships, prefetch_type=pt)
                cp, hp, zp = list(query)

                assertRelationships(cp.relationships, [
                    {'id': r1.id, 'from_person': c.id, 'to_person': h.id},
                    {'id': r2.id, 'from_person': c.id, 'to_person': z.id}])
                assertRelationships(cp.related_to, [
                    {'id': r3.id, 'from_person': h.id, 'to_person': c.id},
                    {'id': r4.id, 'from_person': z.id, 'to_person': c.id}])

                assertRelationships(hp.relationships, [
                    {'id': r3.id, 'from_person': h.id, 'to_person': c.id}])
                assertRelationships(hp.related_to, [
                    {'id': r1.id, 'from_person': c.id, 'to_person': h.id}])

                assertRelationships(zp.relationships, [
                    {'id': r4.id, 'from_person': z.id, 'to_person': c.id}])
                assertRelationships(zp.related_to, [
                    {'id': r2.id, 'from_person': c.id, 'to_person': z.id}])

            with self.assertQueryCount(2):
                query = prefetch(relationships, people, prefetch_type=pt)
                accum = []
                for row in query:
                    accum.append((row.from_person.name, row.to_person.name))
                self.assertEqual(accum, [
                    ('charlie', 'huey'),
                    ('charlie', 'zaizee'),
                    ('huey', 'charlie'),
                    ('zaizee', 'charlie')])

        m = Person.create(name='mickey')
        RC(h, m)

        def assertNames(p, ns):
            self.assertEqual([r.to_person.name for r in p.relationships], ns)

        # Use prefetch to go Person -> Relationship <- Person (PA).
        for pt in SUBQUERY_PREFETCH_TYPES:
            with self.assertQueryCount(3):
                people = (Person
                          .select()
                          .where(Person.name != 'mickey')
                          .order_by(Person.name))
                relationships = Relationship.select().order_by(Relationship.id)
                PA = Person.alias()
                query = prefetch(people, relationships, PA, prefetch_type=pt)
                cp, hp, zp = list(query)
                assertNames(cp, ['huey', 'zaizee'])
                assertNames(hp, ['charlie', 'mickey'])
                assertNames(zp, ['charlie'])

            # User prefetch to go Person -> Relationship+Person (PA).
            with self.assertQueryCount(2):
                people = (Person
                          .select()
                          .where(Person.name != 'mickey')
                          .order_by(Person.name))
                rels = (Relationship
                        .select(Relationship, PA)
                        .join(PA, on=(Relationship.to_person == PA.id))
                        .order_by(Relationship.id))
                query = prefetch(people, rels, prefetch_type=pt)
                cp, hp, zp = list(query)
                assertNames(cp, ['huey', 'zaizee'])
                assertNames(hp, ['charlie', 'mickey'])
                assertNames(zp, ['charlie'])

    def test_prefetch_through_manytomany(self):
        Like.create(note=Note.get(Note.content == 'meow'),
                    person=Person.get(Person.name == 'zaizee'))
        Like.create(note=Note.get(Note.content == 'woof'),
                    person=Person.get(Person.name == 'zaizee'))

        for pt in SUBQUERY_PREFETCH_TYPES:
            with self.assertQueryCount(3):
                people = Person.select().order_by(Person.name)
                notes = Note.select().order_by(Note.content)
                likes = Like.select().order_by(Like.id)
                query = prefetch(people, likes, notes, prefetch_type=pt)
                accum = []
                for person in query:
                    liked_notes = []
                    for like in person.likes:
                        liked_notes.append(like.note.content)
                    accum.append((person.name, liked_notes))

            self.assertEqual(accum, [
                ('huey', ['woof']),
                ('mickey', ['meow']),
                ('zaizee', ['meow', 'woof']),
            ])

    @requires_models(Package, PackageItem)
    def test_prefetch_non_pk_fk(self):
        data = (
            ('101', ('a', 'b')),
            ('102', ('a', 'b')),
            ('103', ()),
            ('104', ('a', 'b', 'c', 'd', 'e')),
        )
        for barcode, items in data:
            Package.create(barcode=barcode)
            for item in items:
                PackageItem.create(package=barcode, name=item)

        for pt in SUBQUERY_PREFETCH_TYPES:
            packages = Package.select().order_by(Package.barcode)
            items = PackageItem.select().order_by(PackageItem.name)

            with self.assertQueryCount(2):
                query = prefetch(packages, items, prefetch_type=pt)
                for package, (barcode, items) in zip(query, data):
                    self.assertEqual(package.barcode, barcode)
                    self.assertEqual([item.name for item in package.items],
                                     list(items))

    def test_prefetch_mark_dirty_regression(self):
        for pt in SUBQUERY_PREFETCH_TYPES:
            people = Person.select().order_by(Person.name)
            query = people.prefetch(Note, NoteItem, prefetch_type=pt)
            for person in query:
                self.assertEqual(person.dirty_fields, [])
                for note in person.notes:
                    self.assertEqual(note.dirty_fields, [])
                    for item in note.items:
                        self.assertEqual(item.dirty_fields, [])

            # Forward-fk population (note.person) must also leave the row
            # clean. The fk setattr would otherwise mark it dirty.
            notes = prefetch(Note.select(), Person.select(), prefetch_type=pt)
            for note in notes:
                self.assertEqual(note.dirty_fields, [])
                self.assertEqual(note.person.dirty_fields, [])


# ===========================================================================
# Multi-reference prefetch (complex FK graphs)
# ===========================================================================

class X(TestModel):
    name = TextField()
class Z(TestModel):
    x = ForeignKeyField(X)
    name = TextField()

class A(TestModel):
    name = TextField()
    x = ForeignKeyField(X)
class B(TestModel):
    name = TextField()
    a = ForeignKeyField(A)
    x = ForeignKeyField(X)
class C(TestModel):
    name = TextField()
    b = ForeignKeyField(B)
    x = ForeignKeyField(X, null=True)

class C1(TestModel):
    name = TextField()
    c = ForeignKeyField(C)
class C2(TestModel):
    name = TextField()
    c = ForeignKeyField(C)


class TestPrefetchMultiRefs(ModelTestCase):
    requires = [X, Z, A, B, C, C1, C2]

    def test_prefetch_multirefs(self):
        x1, x2, x3 = [X.create(name=n) for n in ('x1', 'x2', 'x3')]
        for i, x in enumerate((x1, x2, x3), 1):
            for j in range(i):
                Z.create(x=x, name='%s-z%s' % (x.name, j))

        xs = {x.name: x for x in X.select()}
        xs[None] = None

        data = [
            ('a1',
             'x1',
             ['x1-z0'],
             [
                 ('a1-b1', 'x1', ['x1-z0'], [
                     ('a1-b1-c1', 'x1', ['x1-z0'], [], []),
                 ]),
             ]),
            ('a2',
             'x2',
             ['x2-z0', 'x2-z1'],
             [
                 ('a2-b1', 'x1', ['x1-z0'], [
                     ('a2-b1-c1', 'x1', ['x1-z0'], [], []),
                 ]),
                 ('a2-b2', 'x2', ['x2-z0', 'x2-z1'], [
                     ('a2-b2-c1', 'x2', ['x2-z0', 'x2-z1'], [], []),
                     ('a2-b2-c2', 'x1', ['x1-z0'], [], []),
                     ('a2-b2-cx', None, [], [], []),
                 ]),
             ]),
            ('a3',
             'x3',
             ['x3-z0', 'x3-z1', 'x3-z2'],
             [
                 ('a3-b1', 'x1', ['x1-z0'], [
                     ('a3-b1-c1', 'x1', ['x1-z0'], [], []),
                 ]),
                 ('a3-b2', 'x2', ['x2-z0', 'x2-z1'], [
                     ('a3-b2-c1', 'x2', ['x2-z0', 'x2-z1'], [], []),
                     ('a3-b2-c2', 'x2', ['x2-z0', 'x2-z1'], [], []),
                     ('a3-b2-cx1', None, [], [], []),
                     ('a3-b2-cx2', None, [], [], []),
                     ('a3-b2-cx3', None, [], [], []),
                 ]),
                 ('a3-b3', 'x3', ['x3-z0', 'x3-z1', 'x3-z2'], [
                     ('a3-b3-c1', 'x3', ['x3-z0', 'x3-z1', 'x3-z2'], [], []),
                     ('a3-b3-c2', 'x3', ['x3-z0', 'x3-z1', 'x3-z2'], [], []),
                     ('a3-b3-c3', 'x3', ['x3-z0', 'x3-z1', 'x3-z2'],
                      ['c1-1', 'c1-2', 'c1-3', 'c1-4'],
                      ['c2-1', 'c2-2']),
                 ]),
             ]),
        ]

        for a, ax, azs, bs in data:
            a = A.create(name=a, x=xs[ax])
            for b, bx, bzs, cs in bs:
                b = B.create(name=b, a=a, x=xs[bx])
                for c, cx, czs, c1s, c2s in cs:
                    c = C.create(name=c, b=b, x=xs[cx])
                    for c1 in c1s:
                        C1.create(name=c1, c=c)
                    for c2 in c2s:
                        C2.create(name=c2, c=c)


        AX = X.alias('ax')
        AXZ = Z.alias('axz')
        BX = X.alias('bx')
        BXZ = Z.alias('bxz')
        CX = X.alias('cx')
        CXZ = Z.alias('cxz')

        with self.assertQueryCount(11):
            q = prefetch(A.select().order_by(A.name), *(
                (AX, A), (AXZ, AX),
                (B, A), (BX, B), (BXZ, BX),
                (C, B), (CX, C), (CXZ, CX),
                (C1, C), (C2, C)))

        with self.assertQueryCount(0):
            accum = []
            for a in list(q):
                azs = [z.name for z in a.x.z_set]
                bs = []
                for b in a.b_set:
                    bzs = [z.name for z in b.x.z_set]
                    cs = []
                    for c in b.c_set:
                        czs = [z.name for z in c.x.z_set] if c.x else []
                        c1s = [c1.name for c1 in c.c1_set]
                        c2s = [c2.name for c2 in c.c2_set]
                        cs.append((c.name, c.x.name if c.x else None, czs,
                                   c1s, c2s))

                    bs.append((b.name, b.x.name, bzs, cs))
                accum.append((a.name, a.x.name, azs, bs))

        self.assertEqual(data, accum)


# ===========================================================================
# Multiple FK prefetch with join type control (regression)
# ===========================================================================

class State(TestModel):
    name = TextField()
class Transition(TestModel):
    src = ForeignKeyField(State, backref='sources')
    dest = ForeignKeyField(State, backref='dests')

class TestJoinTypePrefetchMultipleFKs(ModelTestCase):
    requires = [State, Transition]

    def test_join_prefetch_multiple_fks(self):
        s1, s2a, s2b, s3 = [State.create(name=s)
                            for s in ('s1', 's2a', 's2b', 's3')]
        t1 = Transition.create(src=s1, dest=s2a)
        t2 = Transition.create(src=s1, dest=s2b)
        t3 = Transition.create(src=s2a, dest=s3)
        t4 = Transition.create(src=s2b, dest=s3)

        query = State.select().where(State.name != 's3').order_by(State.name)
        transitions = (Transition
                       .select(Transition, State)
                       .join(State, on=Transition.dest)
                       .order_by(Transition.id))
        with self.assertQueryCount(2):
            p = prefetch(query, transitions, prefetch_type=PREFETCH_TYPE.JOIN)
            accum = []
            for row in p:
                accum.append((row.name, row.sources, row.dests,
                              [d.dest.name for d in row.sources],
                              [d.src.name for d in row.dests]))

        self.assertEqual(accum, [
            ('s1', [t1, t2], [], ['s2a', 's2b'], []),
            ('s2a', [t3], [t1], ['s3'], ['s1']),
            ('s2b', [t4], [t2], ['s3'], ['s1'])])


class TestPrefetchErrorPaths(ModelTestCase):
    requires = [Person]

    def test_prefetch_unrelated_model_error(self):
        class Unrelated(TestModel):
            data = TextField()
            class Meta:
                database = self.database

        with self.assertRaises(AttributeError):
            prefetch(Person.select(), Unrelated.select())

    def test_prefetch_empty_subqueries_passthrough(self):
        Person.create(name='huey')
        query = Person.select()
        result = prefetch(query)
        # Should return the query itself (not wrapped).
        self.assertIs(result, query)

    def test_prefetch_materialize_unsupported(self):
        # MATERIALIZE reads keys from already-fetched parents, which the legacy
        # prefetch() engine never holds. It is a with_related()-only strategy.
        with self.assertRaisesRegex(ValueError, 'MATERIALIZE'):
            prefetch(Person.select(), Note,
                     prefetch_type=PREFETCH_TYPE.MATERIALIZE)


# ===========================================================================
# Windowed prefetch: roots delivered in fixed-size windows
# ===========================================================================

class _TrackedCursor(object):
    # Wrap a raw DB-API cursor so tests can see whether it was closed. Raw
    # sqlite3 cursors reject attribute assignment, hence the wrapper.
    def __init__(self, cursor):
        self.cursor = cursor
        self.closed = False

    def close(self):
        self.closed = True
        self.cursor.close()

    def __getattr__(self, attr):
        return getattr(self.cursor, attr)

    def __iter__(self):
        return iter(self.cursor)


class CursorTrackingDatabase(SqliteDatabase):
    def __init__(self, *args, **kwargs):
        super(CursorTrackingDatabase, self).__init__(*args, **kwargs)
        self.cursors = []

    def _execute_cursor(self, sql, params, **cursor_options):
        cursor = super(CursorTrackingDatabase, self)._execute_cursor(
            sql, params, **cursor_options)
        tracked = _TrackedCursor(cursor)
        self.cursors.append(tracked)
        return tracked


class TestPrefetchWindowed(ModelTestCase):
    requires = [Person, Note, NoteItem, Like, Flag, Category]

    def setUp(self):
        super(TestPrefetchWindowed, self).setUp()
        # A tracking database gives each test its own query/cursor record.
        self.db = CursorTrackingDatabase(':memory:')
        self.bind_ctx = self.db.bind_ctx(self.requires)
        self.bind_ctx.__enter__()
        self.addCleanup(self.bind_ctx.__exit__, None, None, None)
        self.db.create_tables(self.requires)

        data = [
            ('huey', [
                ('meow', ['meow-1', 'meow-2']),
                ('purr', []),
                ('hiss', ['hiss-1'])]),
            ('mickey', [
                ('woof', []),
                ('bark', ['bark-1', 'bark-2'])]),
            ('zaizee', [])]
        for name, notes in data:
            person = Person.create(name=name)
            for content, items in notes:
                note = Note.create(person=person, content=content)
                for item in items:
                    NoteItem.create(note=note, content=item)
        Flag.create(note=Note.get(Note.content == 'purr'), is_spam=True)
        Flag.create(note=Note.get(Note.content == 'woof'), is_spam=True)

        def cc(name, parent=None):
            return Category.create(name=name, parent=parent)
        root = cc('root')
        p1 = cc('p1', root)
        cc('p2', root)
        cc('p1-1', p1)

    def _names(self, window):
        return [p.name for p in window]

    def test_window_size_and_query_count(self):
        # Three roots in two windows: one streamed root query plus one
        # relation query per window, no count() and no per-row query.
        self.db.cursors = []
        with self.assertQueryCount(3):
            windows = list(Person.select().order_by(Person.name)
                           .prefetch_windowed(2, Load(Person.notes)))
        self.assertEqual([self._names(w) for w in windows],
                         [['huey', 'mickey'], ['zaizee']])
        self.assertEqual(  # Root query first, relation query per window.
            len(self.db.cursors), 3)

        self.db.cursors = []
        windows = list(Person.select().order_by(Person.name)
                       .prefetch_windowed(2, Load(Person.notes),
                                          Load(Person.likes)))
        # Two relations per window: 1 + 2 * 2.
        self.assertEqual(len(windows), 2)
        self.assertEqual(len(self.db.cursors), 5)

    def test_relations_attached_per_window(self):
        windows = list(Person.select().order_by(Person.name)
                       .prefetch_windowed(2, Load(Person.notes)))
        self.assertEqual(
            [(p.name, sorted(n.content for n in p.notes))
             for w in windows for p in w],
            [('huey', ['hiss', 'meow', 'purr']),
             ('mickey', ['bark', 'woof']),
             ('zaizee', [])])

    def test_window_size_one(self):
        windows = list(Person.select().order_by(Person.name)
                       .prefetch_windowed(1, Load(Person.notes)))
        self.assertEqual([self._names(w) for w in windows],
                         [['huey'], ['mickey'], ['zaizee']])

    def test_no_relations_streams_roots_only(self):
        # Without loads the root query is the only query and still streams
        # in windows rather than returning one materialized list.
        self.db.cursors = []
        windows = list(Person.select().order_by(Person.name)
                       .prefetch_windowed(2))
        self.assertEqual([self._names(w) for w in windows],
                         [['huey', 'mickey'], ['zaizee']])
        self.assertEqual(len(self.db.cursors), 1)

    def test_empty_root_yields_nothing_and_closes_cursor(self):
        self.db.cursors = []
        windows = list(Person.select().where(Person.name == 'nobody')
                       .prefetch_windowed(2, Load(Person.notes)))
        self.assertEqual(windows, [])
        self.assertTrue(self.db.cursors[0].closed)

    def test_root_order_preserved(self):
        windows = list(Person.select().order_by(Person.name.desc())
                       .prefetch_windowed(2))
        self.assertEqual([self._names(w) for w in windows],
                         [['zaizee', 'mickey'], ['huey']])

    def test_multi_level_foreign_keys(self):
        # Parents -> notes -> items and notes -> flags, per window.
        loads = (Load(Person.notes).then(Load(Note.items)),
                 Load(Person.notes).then(Load(Note.flags)))
        windows = list(Person.select().order_by(Person.name)
                       .prefetch_windowed(2, *loads))
        got = [
            (p.name,
             [(n.content,
               sorted(i.content for i in n.items),
               [f.is_spam for f in n.flags])
              for n in sorted(p.notes, key=lambda n: n.content)])
            for w in windows for p in w]
        self.assertEqual(got, [
            ('huey', [('hiss', ['hiss-1'], []),
                      ('meow', ['meow-1', 'meow-2'], []),
                      ('purr', [], [True])]),
            ('mickey', [('bark', ['bark-1', 'bark-2'], []),
                        ('woof', [], [True])]),
            ('zaizee', [])])

    def test_empty_relations_are_empty_lists(self):
        windows = list(Person.select().where(Person.name == 'zaizee')
                       .prefetch_windowed(2, Load(Person.notes)))
        zaizee, = windows[0]
        self.assertEqual(zaizee.notes, [])

    def test_forward_fk(self):
        windows = list(Note.select().order_by(Note.content)
                       .prefetch_windowed(3, Load(Note.person)))
        got = [(n.content, n.person.name) for w in windows for n in w]
        self.assertEqual(got, [
            ('bark', 'mickey'), ('hiss', 'huey'), ('meow', 'huey'),
            ('purr', 'huey'), ('woof', 'mickey')])

    def test_filtered_relation_query(self):
        # A custom relation query narrows rows per window, exactly as with
        # with_related(); excluded rows simply do not attach.
        notes = Note.select().where(Note.content.not_in(('hiss', 'woof')))
        windows = list(Person.select().order_by(Person.name)
                       .prefetch_windowed(2, Load(Person.notes, notes)))
        got = [(p.name, sorted(n.content for n in p.notes))
               for w in windows for p in w]
        self.assertEqual(got, [
            ('huey', ['meow', 'purr']), ('mickey', ['bark']),
            ('zaizee', [])])

    def test_root_query_with_limit_and_join(self):
        # A limited/join-fanned root is streamed as-is: LIMIT caps the whole
        # stream, and duplicate roots arrive as separate instances.
        windows = list(Person.select().order_by(Person.name).paginate(1, 2)
                       .prefetch_windowed(2, Load(Person.notes)))
        self.assertEqual([self._names(w) for w in windows],
                         [['huey', 'mickey']])

        windows = list(Person
                       .select(Person)
                       .join(Note)
                       .where(Note.content << ['meow', 'purr'])
                       .order_by(Person.name)
                       .prefetch_windowed(2, Load(Person.notes)))
        self.assertEqual([self._names(w) for w in windows],
                         [['huey', 'huey']])
        huey = windows[0][0]
        self.assertEqual(sorted(n.content for n in huey.notes),
                         ['hiss', 'meow', 'purr'])

    def test_duplicate_root_rows_are_distinct_instances(self):
        windows = list(Person
                       .select(Person)
                       .join(Note)
                       .where(Note.content << ['meow', 'purr'])
                       .prefetch_windowed(2))
        huey, huey2 = windows[0]
        self.assertIsNot(huey, huey2)

    def test_compound_root_query(self):
        lhs = Person.select().where(Person.name == 'huey')
        rhs = Person.select().where(Person.name == 'mickey')
        windows = list((lhs | rhs).prefetch_windowed(2, Load(Person.notes)))
        self.assertEqual([self._names(w) for w in windows],
                         [['huey', 'mickey']])

    def test_child_identity_not_shared_across_windows(self):
        # A child reachable from roots in two windows is loaded twice: no
        # cross-window Python identity guarantee.
        multi = Note.create(person=Person.get(Person.name == 'huey'),
                            content='shared')
        Like.create(note=multi, person=Person.get(Person.name == 'mickey'))
        Like.create(note=multi, person=Person.get(Person.name == 'zaizee'))

        windows = list(Person.select().order_by(Person.name)
                       .prefetch_windowed(2, Load(Person.likes).then(
                           Load(Like.note))))
        # Window 1 is [huey, mickey], window 2 is [zaizee]; the shared note
        # is reached from a like in each window, but the instances differ.
        self.assertEqual(len(windows), 2)
        first = windows[0][1].likes[0].note
        second = windows[1][0].likes[0].note
        self.assertEqual(first.content, 'shared')
        self.assertEqual(second.content, 'shared')
        self.assertIsNot(first, second)

    def test_window_does_not_rescan_root_query(self):
        # Relation queries filter on the window's keys (MATERIALIZE), not on
        # the full root query: the IN-list holds only that window's pks.
        huey, mickey, zaizee = [Person.get(Person.name == n)
                                for n in ('huey', 'mickey', 'zaizee')]
        self.db.cursors = []
        self.reset_sql_history()
        windows = list(Person.select().order_by(Person.name)
                       .prefetch_windowed(2, Load(Person.notes)))
        self.assertHistory(2, [
            ('SELECT "t1"."id", "t1"."person_id", "t1"."content" '
             'FROM "note" AS "t1" WHERE ("t1"."person_id" IN (?, ?))',
             [huey.id, mickey.id]),
            ('SELECT "t1"."id", "t1"."person_id", "t1"."content" '
             'FROM "note" AS "t1" WHERE ("t1"."person_id" IN (?))',
             [zaizee.id])])

    def test_memory_bounded_to_current_window(self):
        # After consuming all windows and dropping them, every instance --
        # roots and children -- is reclaimable: nothing is retained behind
        # the generator. Child rows are loaded per window and released.
        refs = []
        gen = (Person.select().order_by(Person.name)
               .prefetch_windowed(2, Load(Person.notes)))
        for window in gen:
            for root in window:
                refs.append(weakref.ref(root))
                refs.extend(weakref.ref(note) for note in root.notes)
            del window, root
        del gen
        gc.collect()
        self.assertTrue(all(r() is None for r in refs))

    def test_early_stop_closes_root_cursor(self):
        self.db.cursors = []
        gen = (Person.select().order_by(Person.name)
               .prefetch_windowed(1, Load(Person.notes)))
        iterator = iter(gen)
        first = next(iterator)
        self.assertEqual(self._names(first), ['huey'])
        root_cursor, note_cursor = self.db.cursors[:2]
        self.assertFalse(root_cursor.closed)
        gen.close()
        self.assertTrue(root_cursor.closed)
        self.assertTrue(note_cursor.closed)
        self.assertRaises(StopIteration, next, iterator)

    def test_context_manager_closes_on_early_exit(self):
        self.db.cursors = []
        with (Person.select().order_by(Person.name)
              .prefetch_windowed(1, Load(Person.notes))) as windows:
            for window in windows:
                self.assertEqual(self._names(window), ['huey'])
                break
        root_cursor = self.db.cursors[0]
        self.assertTrue(root_cursor.closed)
        self.assertRaises(sqlite3.ProgrammingError, root_cursor.fetchone)

    def test_early_stop_without_close_after_gc(self):
        # A dropped generator closes its root cursor too, so a skipped
        # close() does not leak the server-side cursor.
        self.db.cursors = []
        gen = (Person.select().order_by(Person.name)
               .prefetch_windowed(1, Load(Person.notes)))
        next(iter(gen))
        root_cursor = self.db.cursors[0]
        del gen
        gc.collect()
        self.assertTrue(root_cursor.closed)

    def test_root_query_not_mutated(self):
        query = Person.select().order_by(Person.name)
        list(query.prefetch_windowed(2, Load(Person.notes)))
        # The caller's query was cloned, so it is still runnable on its own.
        self.assertEqual(len(list(query)), 3)

    def test_window_size_validation(self):
        for bad in (0, -1, 1.5, None, '2'):
            self.assertRaises(ValueError, prefetch_windowed,
                              Person.select(), bad, Load(Person.notes))

    def test_requires_model_select(self):
        # Non-model row types and a root query that already carries a
        # with_related() load tree are both rejected up-front.
        self.assertRaises(ValueError, prefetch_windowed,
                          Note.select().dicts(), 2)
        self.assertRaises(ValueError, prefetch_windowed,
                          Note.select().tuples(), 2)
        self.assertRaises(ValueError, prefetch_windowed,
                          Note.select().with_related(Load(Note.person)), 2)

    def test_single_iteration(self):
        gen = Person.select().prefetch_windowed(2)
        list(gen)
        self.assertRaises(ValueError, list, gen)

    def test_bare_relations_and_loads_mixed(self):
        windows = list(Person.select().order_by(Person.name)
                       .prefetch_windowed(2, Person.notes,
                                          Load(Person.likes)))
        self.assertEqual(
            [(p.name, len(p.notes), len(p.likes)) for w in windows for p in w],
            [('huey', 3, 0), ('mickey', 2, 0), ('zaizee', 0, 0)])

    @requires_models(Category)
    def test_self_referential_relation(self):
        def cc(name, parent=None):
            return Category.create(name=name, parent=parent)
        c = cc('c')
        p1 = cc('p1', c)
        p2 = cc('p2', c)
        cc('p1-1', p1)
        cc('p2-1', p2)
        cc('p2-2', p2)

        # A self-referential backref needs no alias under windowed prefetch:
        # the relation filters on the window's keys, so the child query can
        # never correlate with itself.
        windows = list(Category.select().order_by(Category.name)
                       .prefetch_windowed(2, Load(Category.children)))
        got = [(cat.name, sorted(ch.name for ch in cat.children))
               for w in windows for cat in w]
        self.assertEqual(got, [
            ('c', ['p1', 'p2']), ('p1', ['p1-1']), ('p1-1', []),
            ('p2', ['p2-1', 'p2-2']), ('p2-1', []), ('p2-2', [])])

        # Nested self-referential loads (parents -> children -> grandchildren).
        windows = list(Category.select().where(Category.name == 'c')
                       .prefetch_windowed(2, Load(Category.children).then(
                           Load(Category.children))))
        c_root, = windows[0]
        self.assertEqual(
            {ch.name: sorted(g.name for g in ch.children)
             for ch in c_root.children},
            {'p1': ['p1-1'], 'p2': ['p2-1', 'p2-2']})

    @requires_models(Category)
    def test_self_referential_alias_parent(self):
        root = Category.create(name='root')
        Category.create(name='a', parent=root)
        Category.create(name='b', parent=root)
        CA = Category.alias()
        windows = list(CA.select().where(CA.name == 'root')
                       .prefetch_windowed(2, Load(Category.children)))
        root_alias, = windows[0]
        self.assertEqual(sorted(c.name for c in root_alias.children),
                         ['a', 'b'])

    def test_per_parent_limit_per_window(self):
        # Each window ranks its own children; the ranking CTE is rebuilt per
        # window on that window's keys and keeps the top-N of every parent.
        query = (Note.select().order_by(Note.content.desc())
                 .where(Note.content != 'woof'))
        windows = list(Person.select().order_by(Person.name)
                       .prefetch_windowed(2, Load(Person.notes, query,
                                                  per_parent=2)))
        got = [(p.name, sorted(n.content for n in p.notes))
               for w in windows for p in w]
        self.assertEqual(got, [
            ('huey', ['meow', 'purr']), ('mickey', ['bark']),
            ('zaizee', [])])

    @requires_models(Package, PackageItem)
    def test_non_pk_foreign_key(self):
        # The windowed IN-list is built from the fk's rel_field (barcode),
        # not the pk, matching with_related() MATERIALIZE semantics.
        data = (('101', ('a', 'b')), ('102', ()), ('104', ('c',)))
        for barcode, items in data:
            Package.create(barcode=barcode)
            for name in items:
                PackageItem.create(package=barcode, name=name)

        self.reset_sql_history()
        windows = list(Package.select().order_by(Package.barcode)
                       .prefetch_windowed(2, Load(Package.items)))
        got = [(p.barcode, sorted(i.name for i in p.items))
               for w in windows for p in w]
        self.assertEqual(got, [('101', ['a', 'b']), ('102', []),
                               ('104', ['c'])])
        self.assertEqual(
            self.history[1].msg[0],
            'SELECT "t1"."id", "t1"."name", "t1"."package_id" '
            'FROM "package_item" AS "t1" '
            'WHERE ("t1"."package_id" IN (?, ?))')
        self.assertEqual(self.history[1].msg[1], ['101', '102'])

    def test_database_kwarg_propagates(self):
        alt = CursorTrackingDatabase(':memory:')
        with alt.bind_ctx(self.requires):
            alt.create_tables(self.requires)
            alt_person = Person.create(name='alt')
            Note.create(person=alt_person, content='alt-note')
            windows = list(Person.select().order_by(Person.name)
                           .prefetch_windowed(1, Load(Person.notes),
                                              database=alt))
            root, = windows[0]
            self.assertEqual(root.name, 'alt')
            self.assertEqual([n.content for n in root.notes], ['alt-note'])
            self.assertTrue(alt.cursors)
        alt.close()

    def test_relation_strategy_overridden_to_materialize(self):
        # Whatever strategy the caller names, windowed scoping reads the
        # window's keys: the supplied nodes are not mutated.
        load = Load(Person.notes, strategy=PREFETCH_TYPE.JOIN)
        self.reset_sql_history()
        windows = list(Person.select().order_by(Person.name)
                       .prefetch_windowed(2, load))
        self.assertEqual(load._strategy, PREFETCH_TYPE.JOIN)
        self.assertEqual(
            [(p.name, len(p.notes)) for w in windows for p in w],
            [('huey', 3), ('mickey', 2), ('zaizee', 0)])
        self.assertIn('IN (?, ?)', self.history[1].msg[0])
