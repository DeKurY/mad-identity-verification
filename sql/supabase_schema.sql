-- Secure-by-default Supabase schema for the coursework KYC demo.
--
-- This file creates the tables and ENABLES row-level security (RLS). It does not
-- add permissive public policies. With the anon key, reads/writes will be blocked
-- until you add proper authentication and policies for your use case.
--
-- For a quick classroom demo with synthetic data only, see:
--   sql/supabase_schema_demo_insecure.sql

create extension if not exists pgcrypto;

create table if not exists public.identity_profiles (
    id uuid primary key default gen_random_uuid(),
    full_name text not null,
    last_name text,
    first_name text,
    middle_name text,
    birth_date date,
    passport_series text not null,
    passport_number text not null,
    reference_face_embedding jsonb not null,
    reference_image_path text,
    embedding_model text not null default 'Facenet512',
    detector_backend text not null default 'retinaface',
    embedding_dim integer not null default 512,
    created_at timestamptz not null default now(),
    unique (passport_series, passport_number)
);

create table if not exists public.verification_attempts (
    id uuid primary key default gen_random_uuid(),
    identity_id uuid references public.identity_profiles(id) on delete set null,
    input_passport_data jsonb,
    parser_debug jsonb,
    ocr_items jsonb,
    passport_photo_embedding jsonb,
    selfie_embedding jsonb,
    passport_reference_distance double precision,
    selfie_passport_distance double precision,
    selfie_reference_distance double precision,
    face_accept_threshold double precision,
    face_review_threshold double precision,
    data_match_score double precision,
    data_verified boolean,
    passport_photo_verified boolean,
    selfie_verified boolean,
    final_decision text check (final_decision in ('ACCEPT', 'REVIEW', 'REJECT')),
    passport_photo_path text,
    selfie_path text,
    error_message text,
    created_at timestamptz not null default now()
);

alter table public.identity_profiles enable row level security;
alter table public.verification_attempts enable row level security;

comment on table public.identity_profiles is 'Stores identity profiles. Contains passport data and biometric embeddings; keep RLS enabled for non-demo use.';
comment on table public.verification_attempts is 'Stores verification attempts. Debug columns can contain OCR text and biometric data; keep them restricted.';
