# Model cache design

[ModelCache](../src/libmlvc/inference/model_cache.hpp) stores compiled models on disk for reuse across starts. It is inference-engine agnostic: backends compile and load models; the cache manages the files.

Each key identifies one entry: a directory containing compiled model files.

## Public API

| Method | Purpose |
| --- | --- |
| [Create](#create) | Create a cache instance for an entries directory. |
| [Get](#get) | Validate and return an existing entry's directory without compiling. |
| [GetOrCreate](#getorcreate) | Return a validated entry directory, compiling through the supplied callback if the entry is missing. |
| [Cleanup](#cleanup) | Remove leftovers and apply retention and size policies. |

Keys are case-insensitive, nonempty strings containing only ASCII letters, digits, `_`, or `-`. [GetOrCreate](#getorcreate) also requires a nonempty compile callback. Invalid keys or callbacks return an argument error.

## Contract

- Multiple applications can share the same cache directory, as can multiple cache instances within one process. Concurrent compilation of the same missing entry across processes must be handled gracefully. The first successful publication wins; competing publishers validate that entry before reuse. Losing the publication race is not itself an error.
- With caching enabled, cache-capable models load only from cache. Cache errors fail model loading; there is no fallback to in-memory compilation.
- [Get](#get) and [GetOrCreate](#getorcreate) never repair or remove existing published entries. Failed initial validation does not refresh access time. Only the [cleanup phases](#cleanup) remove published entries. A removed entry can later be recreated by [GetOrCreate](#getorcreate).

## Storage

The caller chooses the entries directory. For an entries directory named `models`, the layout is:

```text
<cacheDir>
  models
    v2_<environment>_<model>
      <compiled files>
      mlvc_manifest.json
      last_access.time
  models.staging
  models.trash
```

Staging and trash are sibling directories named by appending `.staging` and `.trash` to the entries-directory path.

The cache prepends `v2_` to normalized keys but does not interpret them. Callers must encode model identity and compilation dependencies, such as execution-provider and driver versions. The example uses an environment/model pair.

The cache root must be reserved for MLVC cache data: cleanup may remove directories under any of the three cache folders.

### On-disk format

- **`mlvc_manifest.json`**: a JSON manifest of at most **1 MiB**, with a positive integer version and a nonempty file list. Paths are nonempty, entry-relative UTF-8 strings. They use `/` separators and cannot contain embedded NULs, a root path, `.` components, or `..` components. Sizes are nonnegative integers in bytes.
- **`last_access.time`**: a signed 64-bit little-endian integer representing seconds since 1970. It is the only file the cache updates after publication, in place without truncation or recreation.

```json
{
  "version": 1,
  "files": [
    { "path": "model.onnx", "size": 985 }
  ]
}
```

Writers emit version 1. Readers accept any positive version, require the base fields above, and ignore unknown fields. Extensions must preserve existing fields and meanings; incompatible changes require a new cache-key prefix, not just a higher manifest version.

On Windows, readers convert manifest paths to native separators for filesystem access, including under extended-length (`\\?\`) entries paths. The on-disk paths still use `/` separators.

## API behavior

### Create

`ModelCache::Create` rejects empty paths and reports resolution errors. It lexically normalizes the entries-directory path and makes it absolute at construction, so working-directory changes cannot redirect entries, staging, or trash.

Directories are created lazily; construction does not guarantee writability. It attempts best-effort [leftover cleanup](#cleanup): entries with missing or unreadable access times, abandoned staging directories, and trash.

### Get

1. **Validate**: check the manifest and each listed file's size. Return an error if the entry is missing or any validation check fails. Causes include an unreadable or invalid manifest, a missing listed file, or a size mismatch.
2. **Refresh access time**: update the access time. If this fails, log a warning and continue.
3. **Revalidate and return**: repeat validation to detect cleanup races. Return the directory if the entry is valid, or an error otherwise.

The method never compiles a model or retries a missing entry.

### GetOrCreate

1. **Find or create**: take the process-wide per-key lock and check the final entry path. If present, skip to top-level step 2; otherwise, hold the lock through:

    1. **Prepare staging**: create a randomly named staging directory, independent of the key, and its `last_access.time` file before compilation so cleanup recognizes recent work. Return an error if either cannot be created.
    2. **Compile**: invoke the callback with the staging directory and propagate errors. Reject a missing or unreadable timestamp afterward: cleanup may have removed staging, and the compiler may have only partly recreated it.
    3. **Write the manifest and flush**: list compiled files; return an error if none exist. Reject serialized manifests over **1 MiB** before writing, then write the manifest and flush staged files. Return an error without publishing on any filesystem failure.
    4. **Publish**: rename the timestamped staging directory to the final path. If a competing entry exists after a rename failure, discard staging and validate that entry in top-level step 2. Retry other rename failures for about one second to tolerate file locks. Return an error if publication still fails.

2. **Validate and refresh**: release the lock, validate and refresh as in [Get](#get), and return the directory on success.
3. **Retry once if missing**: cleanup does not use the creation lock and may remove an entry during validation. If validation fails and the final entry is missing, repeat top-level steps 1 and 2 once. Return an error for an existing invalid entry or a failed retry. Return creation errors immediately.

**Concurrency**: creation locks are shared process-wide by normalized entry name, not path. Creation for the same key is serialized across roots and aliases; physically distinct roots keep separate files. Different keys use independent locks. [Get](#get) and [Cleanup](#cleanup) do not acquire these locks.

**Staging cleanup**: an attempt's staging directory is removed on a best-effort basis if creation fails or another publisher wins.

### Cleanup

Construction runs only leftover cleanup. Explicit calls run all three phases below; lookups never trigger cleanup. Retention and size limits are caller-configurable.

1. **Remove leftovers**:
    - **Published entries**: remove directories without a readable `last_access.time`, even if compiled files validate.
    - **Staging**: remove directories with timestamps at least **1 hour** in the past or future. If `last_access.time` is missing or unreadable, use directory modification time. Leave the directory if neither time is readable.
    - **Trash**: delete directories already moved to trash.
2. **Expire entries**: remove published entries whose access time is at least **14 days** old by default, regardless of validation status.
3. **Apply the soft size limit**: if the measured total size of remaining published entries exceeds the configured limit (**512 MiB** by default):
    1. Validate the remaining entries without refreshing their access timestamps.
    2. Order invalid entries before valid ones. In each group, select entries with timestamps furthest from now, past or future.
    3. Remove candidates in order until the measured size is at or below the limit, or all candidates are tried. No recent-use exemption applies; failed removals or concurrent publications can leave the cache above the limit.

**Removal**: reread access time before moving an entry or staging directory; skip it if the read result changed. This check is not atomic with the rename. Move to trash before attempting deletion. Failed moves leave the original in place; failed deletions leave it in trash for later cleanup.

Linked cache folders and candidate directories are skipped. A linked trash directory blocks removals.

## Current limitations

Ordered by practical impact; occurrence rates are not quantified.

- **Accepted deferred-loading risk**: by default, a prepared entry becomes eligible for expiry after **14 days without recorded cache access**. If cleanup removes it before a deferred consumer first uses it, the read-only [Get](#get) lookup fails. Compilation inputs are discarded to save memory, so recovery may require reinitialization from the original bundles or a process restart.
- **Persistent load failures**: manifest and file-size checks do not establish content integrity or loadability. Load failures do not invalidate entries. Repeated successful access-time updates can prevent time-based expiry, though size eviction still applies. A new environment/model key avoids reusing the entry.
- **Incomplete cleanup accounting**: filesystem errors can produce partial directory listings or byte counts without an error status. Cleanup may miss entries or underestimate usage. Later passes may recover once the errors clear.

## Migration from older versions

**Upgrade and downgrade**: legacy entries lack a manifest and the `v2_` prefix. Current lookups never reuse them; [GetOrCreate](#getorcreate) compiles separate entries as needed. Older versions keep their keys and may recompile after cleanup.

**Legacy-entry cleanup**: both formats count toward the budget. Legacy entries follow the same leftover and expiry rules but are prioritized as invalid during size eviction because they lack manifests.

**Compatibility requirements**: older cleanup scans entries of both formats and removes directories whose access time is missing, unreadable, or older than the default **60-day** retention period. While older versions may share the cache:

- Preserve the `last_access.time` filename and signed 64-bit little-endian seconds-since-1970 format.
- Keep staging and trash outside the entries directory (`models` in the example layout).

**Remaining risks**: older cleanup deletes in place without rechecking access times, and older timestamp writes truncate the file. Concurrency can expose partially deleted entries or make current cleanup misclassify live legacy entries as untimed leftovers.
