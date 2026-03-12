# Kernel Boot Validation

## QEMU Invocation

```bash
qemu-system-riscv64 \
  -machine virt \
  -kernel external/linux-6.9/arch/riscv/boot/Image \
  -initrd external/initramfs.cpio.gz \
  -nographic \
  -append "console=ttyS0 rdinit=/init" \
  -m 256M \
  -smp 1
```

## Init Binary

Minimal static RISC-V binary that prints `USERSPACE_OK\n` and calls
`reboot(LINUX_REBOOT_CMD_POWER_OFF)`.

## GCC Baseline

GCC-compiled kernel boots successfully — `USERSPACE_OK` confirmed in
serial output.
