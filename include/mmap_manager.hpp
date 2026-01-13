#ifndef MMAP_MANAGER_H
#define MMAP_MANAGER_H

#include "params.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>

#if defined(__linux__) || defined(__APPLE__)
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#else
  #error "mmap_manager.hpp supports POSIX (Linux/macOS) only."
#endif

namespace mmap_manager {

// -----------------------------
// Payload (packed, fixed layout)
// -----------------------------
#pragma pack(push, 1)
struct LogData {
  float t              =  0.0f;  // timestamp [sec]
  float pos_d[3]       = {0.0f}; // desired position [m]
  float pos[3]         = {0.0f}; // current position [m]
  float rpy[3]         = {0.0f}; // current attitude [rad]
  float rpy_raw[3]     = {0.0f}; // desired attitude from position ctrl [rad]
  float rpy_d[3]       = {0.0f}; // desired attitude reconstructed (R_d) [rad]
  float tau_d[3]       = {0.0f}; // desired torque (att ctrl) [N.m]
  float tau_off[2]     = {0.0f}; // cot-offset torque [N.m] (x,y)
  float tau_thrust[2]  = {0.0f}; // thrust-diff torque [N.m] (x,y)
  float tilt_rad[4]    = {0.0f}; // per-rotor tilt command [rad]
  float f_thrust[4]    = {0.0f}; // per-rotor thrust command [N]
  float f_total        =  0.0f;  // desired collective thrust [N]
  float r_cot[2]       = {0.0f}; // current CoT position [m] (x,y)
  float r_cot_cmd[2]   = {0.0f}; // optimal CoT command [m] (x,y)
  float solve_ms       =  0.0f;  // acados solve time [ms]
  int32_t solve_status = -1;     // solver status
};
#pragma pack(pop)

// 152 bytes with the layout above
static_assert(sizeof(LogData) == 152, "LogData size changed. Update Python reader offsets.");


// -----------------------------
// MMap header + ring buffer slot
// - Slot must be 8-byte aligned to avoid SIGBUS on atomic u64 access.
// -----------------------------
#pragma pack(push, 1)
struct MMapHeader {
  char     magic[8];      // "STRLOG2\0"
  uint32_t version;       // 2
  uint32_t header_size;   // sizeof(MMapHeader)
  uint32_t capacity;      // number of slots
  uint32_t slot_size;     // sizeof(Slot)
  uint64_t write_count;   // monotonically increasing
  uint64_t start_time_ns; // optional (0 ok)
  uint8_t  reserved[24];  // pad to 64 bytes
};
#pragma pack(pop)

static_assert(sizeof(MMapHeader) == 64, "MMapHeader must be 64 bytes.");

// Slot = seq(8) + LogData(152) = 160 (already multiple of 8)
struct alignas(8) Slot {
  uint64_t seq;  // seqlock counter (odd=writing, even=stable)
  LogData  data; // payload
};

static_assert(alignof(Slot) == 8, "Slot alignment must be 8.");
static_assert(sizeof(Slot) == 160, "Slot size must be 160 bytes.");

static constexpr uint32_t k_Sec = 10;
static constexpr uint32_t k_Cap = static_cast<uint32_t>(param::CTRL_HZ) * k_Sec;

inline uint64_t atomic_load_u64(const uint64_t* p) {
  return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}
inline void atomic_store_u64(uint64_t* p, uint64_t v) {
  __atomic_store_n(p, v, __ATOMIC_RELEASE);
}


// -----------------------------
// Writer (SPSC producer-side)
// - No locks in control loop.
// -----------------------------
class MMapLogger {
public:
  explicit MMapLogger(const std::string& path = "/tmp/strider_log.mmap", bool reset = true)
  : path_(path), reset_(reset) {}

  ~MMapLogger() { close(); }

  MMapLogger(const MMapLogger&) = delete;
  MMapLogger& operator=(const MMapLogger&) = delete;

  void open() {
    if (opened_) return;

    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd_ < 0) throw std::runtime_error("mmap_logger: open() failed: " + path_);

    map_size_ = sizeof(MMapHeader) + static_cast<size_t>(k_Cap) * sizeof(Slot);

    if (::ftruncate(fd_, static_cast<off_t>(map_size_)) != 0) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("mmap_logger: ftruncate() failed");
    }

    void* p = ::mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("mmap_logger: mmap() failed");
    }

    base_   = static_cast<uint8_t*>(p);
    header_ = reinterpret_cast<MMapHeader*>(base_);
    slots_  = reinterpret_cast<Slot*>(base_ + sizeof(MMapHeader));

    if (reset_) {
      std::memset(base_, 0, map_size_);

      const char kMagic[8] = {'S','T','R','L','O','G','2','\0'};
      std::memcpy(header_->magic, kMagic, 8);
      header_->version       = 2;
      header_->header_size   = static_cast<uint32_t>(sizeof(MMapHeader));
      header_->capacity      = k_Cap;
      header_->slot_size     = static_cast<uint32_t>(sizeof(Slot));
      header_->start_time_ns = 0;

      // write_count must be stored last
      atomic_store_u64(&header_->write_count, 0);
    } else {
      // Validate basic compatibility; if mismatch, reset.
      if (std::strncmp(header_->magic, "STRLOG2", 7) != 0 ||
          header_->version != 2 ||
          header_->capacity != k_Cap ||
          header_->slot_size != sizeof(Slot) ||
          header_->header_size != sizeof(MMapHeader)) {
        std::memset(base_, 0, map_size_);
        const char kMagic[8] = {'S','T','R','L','O','G','2','\0'};
        std::memcpy(header_->magic, kMagic, 8);
        header_->version       = 2;
        header_->header_size   = static_cast<uint32_t>(sizeof(MMapHeader));
        header_->capacity      = k_Cap;
        header_->slot_size     = static_cast<uint32_t>(sizeof(Slot));
        header_->start_time_ns = 0;
        atomic_store_u64(&header_->write_count, 0);
      }
    }

    opened_ = true;
  }

  void close() {
    if (!opened_) return;

    ::msync(base_, map_size_, MS_ASYNC);
    ::munmap(base_, map_size_);
    base_ = nullptr;
    header_ = nullptr;
    slots_ = nullptr;
    map_size_ = 0;

    ::close(fd_);
    fd_ = -1;

    opened_ = false;
  }

  inline void push(const LogData& x) {
    if (!opened_) open();

    const uint64_t wc = atomic_load_u64(&header_->write_count);
    const uint32_t idx = static_cast<uint32_t>(wc % k_Cap);
    Slot* s = &slots_[idx];

    // Seqlock write: seq odd -> memcpy -> seq even
    const uint64_t seq0 = atomic_load_u64(&s->seq);
    atomic_store_u64(&s->seq, seq0 + 1); // mark writing (odd)

    std::memcpy(&s->data, &x, sizeof(LogData));

    atomic_store_u64(&s->seq, seq0 + 2); // done (even)
    atomic_store_u64(&header_->write_count, wc + 1);
  }

  uint64_t write_count() const {
    if (!opened_) return 0;
    return atomic_load_u64(&header_->write_count);
  }

  static constexpr uint32_t capacity() { return k_Cap; }
  static constexpr uint32_t header_size() { return sizeof(MMapHeader); }
  static constexpr uint32_t slot_size() { return sizeof(Slot); }

private:
  std::string path_;
  bool reset_ = true;

  bool opened_ = false;
  int fd_ = -1;
  size_t map_size_ = 0;

  uint8_t* base_ = nullptr;
  MMapHeader* header_ = nullptr;
  Slot* slots_ = nullptr;
};

} // namespace mmap_manager

#endif // MMAP_MANAGER_H