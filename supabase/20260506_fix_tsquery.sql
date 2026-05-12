-- ============================================================
-- Migration: fix hybrid_search to use plainto_tsquery throughout
-- Apply to a running database:
--   docker exec -i edintech-postgres psql -U edintech edintechrag \
--     < supabase/20260506_fix_tsquery.sql
-- ============================================================

create or replace function hybrid_search(
    query_text             text,
    query_embedding        vector(1024),
    match_count            int,
    rrf_k                  int default 60,
    filter_equipment_id    text default null,
    filter_document_category document_category default null,
    filter_file_type       file_type default null,
    filter_location        text default null
)
returns table (
    chunk_id          bigint,
    chunk_document_id bigint,
    chunk_content     text,
    chunk_metadata    jsonb,
    doc_filename      text,
    doc_category      document_category,
    doc_equipment_id  text,
    doc_location      text,
    rrf_score         float
)
language plpgsql
as $$
declare
    v_limit int;
begin
    v_limit := least(match_count * 3, 500);

    return query
    with text_matches as (
        select
            c.id                                                                          as tm_id,
            c.document_id                                                                 as tm_doc_id,
            c.content                                                                     as tm_content,
            c.metadata                                                                    as tm_metadata,
            d.document_category                                                           as tm_category,
            d.equipment_id                                                                as tm_equip,
            d.location                                                                    as tm_location,
            1.0 / (rrf_k + row_number() over (
                order by ts_rank(c.fts, plainto_tsquery('english', query_text)) desc
            ))                                                                            as tm_rank
        from chunks c
        inner join documents d on d.id = c.document_id
        where c.fts @@ plainto_tsquery('english', query_text)
          and (filter_equipment_id      is null or d.equipment_id      = filter_equipment_id)
          and (filter_document_category is null or d.document_category = filter_document_category)
          and (filter_file_type         is null or d.file_type         = filter_file_type)
          and (filter_location          is null or d.location          = filter_location)
        order by ts_rank(c.fts, plainto_tsquery('english', query_text)) desc
        limit v_limit
    ),
    vector_matches as (
        select
            c.id                                                                          as vm_id,
            c.document_id                                                                 as vm_doc_id,
            c.content                                                                     as vm_content,
            c.metadata                                                                    as vm_metadata,
            d.document_category                                                           as vm_category,
            d.equipment_id                                                                as vm_equip,
            d.location                                                                    as vm_location,
            1.0 / (rrf_k + row_number() over (
                order by c.embedding <=> query_embedding
            ))                                                                            as vm_rank
        from chunks c
        inner join documents d on d.id = c.document_id
        where c.embedding is not null
          and (filter_equipment_id      is null or d.equipment_id      = filter_equipment_id)
          and (filter_document_category is null or d.document_category = filter_document_category)
          and (filter_file_type         is null or d.file_type         = filter_file_type)
          and (filter_location          is null or d.location          = filter_location)
        order by c.embedding <=> query_embedding
        limit v_limit
    ),
    combined as (
        select
            am.am_id,
            am.am_doc_id,
            am.am_content,
            am.am_metadata,
            am.am_category,
            am.am_equip,
            am.am_location,
            sum(am.am_rank) as am_score
        from (
            select
                tm.tm_id       as am_id,
                tm.tm_doc_id   as am_doc_id,
                tm.tm_content  as am_content,
                tm.tm_metadata as am_metadata,
                tm.tm_category as am_category,
                tm.tm_equip    as am_equip,
                tm.tm_location as am_location,
                tm.tm_rank     as am_rank
            from text_matches tm
            union all
            select
                vm.vm_id       as am_id,
                vm.vm_doc_id   as am_doc_id,
                vm.vm_content  as am_content,
                vm.vm_metadata as am_metadata,
                vm.vm_category as am_category,
                vm.vm_equip    as am_equip,
                vm.vm_location as am_location,
                vm.vm_rank     as am_rank
            from vector_matches vm
        ) am
        group by am.am_id, am.am_doc_id, am.am_content, am.am_metadata,
                 am.am_category, am.am_equip, am.am_location
        order by am_score desc
        limit match_count
    )
    select
        comb.am_id           as chunk_id,
        comb.am_doc_id       as chunk_document_id,
        comb.am_content      as chunk_content,
        comb.am_metadata     as chunk_metadata,
        d.filename           as doc_filename,
        comb.am_category     as doc_category,
        comb.am_equip        as doc_equipment_id,
        comb.am_location     as doc_location,
        comb.am_score::float as rrf_score
    from combined comb
    inner join documents d on d.id = comb.am_doc_id;
end;
$$;
