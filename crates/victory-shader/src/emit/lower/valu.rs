//! Vector ALU and comparison instructions lowered to DXIL.

use super::{LANE_MASK, Lower, MaskData, Val, Why, inline_f16};
use crate::emit::bitcode::{Attr, Bin, Cast, V, pred};
use crate::emit::shader::flags;
use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::ir::{EXEC, Inst, Reg};

impl Lower<'_> {
    pub(super) fn valu(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if self.x.res.waterfall.compares.contains(&(b, i)) {
            // The first lane's dwords of a waterfall loop are each lane's
            // own (see `resource::Waterfall`).
            self.lane(inst, Some(true));
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("v_cmp_").or(name.strip_prefix("v_cmpx_")) {
            let r = if v.dpp.is_some() {
                Err("a compare with DPP")
            } else {
                self.compare(inst, rest)
            };
            if let Err(why) = r {
                for &(_, w) in &inst.writes {
                    self.lane_fail.insert(w, why);
                }
            }
            return r;
        }
        let mut guard = self.guard(b, i)?;
        if let Some(d) = v.dpp {
            let (x, enable) = self.dpp(inst, d)?;
            guard = match (guard, enable) {
                (Some(g), Some(e)) => Some(self.s.m.bin(self.cur, Bin::And, g, e)),
                (g, e) => g.or(e),
            };
            self.src0 = Some(x);
        }
        let r = self.valu_op(inst, guard);
        self.src0 = None;
        r
    }

    /// DPP: src0 as read from another lane, and which lanes write their
    /// result (none: all).
    fn dpp(&mut self, inst: &Inst, dpp: Dpp) -> Result<(V, Option<V>), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else {
            return Err("a DPP operand");
        };
        let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("a DPP operand")?)?;
        // Quad permutations and DPP8 stay in groups the pixel layout shares.
        let quad = matches!(dpp, Dpp::Row { ctrl: 0..=0xFF, .. });
        let mesh_quad = self.s.mesh.is_some() && quad;
        if !mesh_quad {
            self.cross_lane(quad)?;
        }
        let c = self.cur;
        let lane = self.lane_index();
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (sixteen_mask, low) = (k(self, !15), k(self, 15));
        let row = self.s.m.bin(c, Bin::And, lane, sixteen_mask);
        let pos = self.s.m.bin(c, Bin::And, lane, low);
        // Source lane, and whether it lies in the row.
        let (src, inside, masks, zero) = match dpp {
            Dpp::Lanes { sel, .. } => {
                let (seven, not7, three) = (k(self, 7), k(self, !7), k(self, 3));
                let p = self.s.m.bin(c, Bin::And, lane, seven);
                let sh = self.s.m.bin(c, Bin::Mul, p, three);
                let sel = k(self, sel);
                let pick = self.s.m.bin(c, Bin::LShr, sel, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, seven);
                let base = self.s.m.bin(c, Bin::And, lane, not7);
                (self.s.m.bin(c, Bin::Or, base, pick), None, None, false)
            }
            Dpp::Row {
                ctrl,
                row_mask,
                bank_mask,
                bound_ctrl,
                ..
            } => {
                let n = u32::from(ctrl & 15);
                let (src, inside) = match ctrl {
                    0..=0xFF => {
                        let (three, one, not3) = (k(self, 3), k(self, 1), k(self, !3));
                        let p = self.s.m.bin(c, Bin::And, lane, three);
                        let sh = self.s.m.bin(c, Bin::Shl, p, one);
                        let perm = k(self, u32::from(ctrl));
                        let pick = self.s.m.bin(c, Bin::LShr, perm, sh);
                        let pick = self.s.m.bin(c, Bin::And, pick, three);
                        let base = self.s.m.bin(c, Bin::And, lane, not3);
                        (self.s.m.bin(c, Bin::Or, base, pick), None)
                    }
                    0x101..=0x10F | 0x111..=0x11F => {
                        let nn = k(self, n);
                        let left = ctrl < 0x110;
                        let (op, p) = if left {
                            (Bin::Add, pred::IULT)
                        } else {
                            (Bin::Sub, pred::IUGE)
                        };
                        let s = self.s.m.bin(c, op, lane, nn);
                        // Left: pos + n < 16; right: pos >= n.
                        let bound = if left { k(self, 16 - n) } else { nn };
                        (s, Some(self.s.m.cmp(c, p, pos, bound)))
                    }
                    0x121..=0x12F => {
                        let nn = k(self, n);
                        let t = self.s.m.bin(c, Bin::Sub, pos, nn);
                        let t = self.s.m.bin(c, Bin::And, t, low);
                        (self.s.m.bin(c, Bin::Or, row, t), None)
                    }
                    0x140 | 0x141 => {
                        let m = k(self, if ctrl == 0x140 { 15 } else { 7 });
                        let t = self.s.m.bin(c, Bin::Xor, lane, m);
                        (t, None)
                    }
                    0x150..=0x15F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Or, row, nn), None)
                    }
                    0x160..=0x16F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Xor, lane, nn), None)
                    }
                    _ => return Err("a DPP lane pattern"),
                };
                let masks = (row_mask != 0xF || bank_mask != 0xF).then_some((row_mask, bank_mask));
                (src, inside, masks, bound_ctrl)
            }
        };
        let mut r = if mesh_quad || self.helpers_read(quad) {
            self.quad_pick(x, src)
        } else {
            self.read_lane(x, src)
        };
        let mut enable = None;
        if let Some(inside) = inside {
            if zero {
                let z = k(self, 0);
                r = self.s.m.select(c, inside, r, z);
            } else {
                enable = Some(inside);
            }
        }
        if let Some((rows, banks)) = masks {
            // Row (16 lanes) and bank (4 lanes of a row) write masks. Rows
            // are the console's: split in halves, the upper half's host rows
            // 0 and 1 are console rows 2 and 3.
            let (four, two, three) = (k(self, 4), k(self, 2), k(self, 3));
            let row_lane = if self.split {
                self.console_lane().0
            } else {
                lane
            };
            let ri = self.s.m.bin(c, Bin::LShr, row_lane, four);
            let ri = self.s.m.bin(c, Bin::And, ri, three);
            let bi = self.s.m.bin(c, Bin::LShr, lane, two);
            let bi = self.s.m.bin(c, Bin::And, bi, three);
            let (rm, bm) = (k(self, u32::from(rows)), k(self, u32::from(banks)));
            let rb = self.s.m.bin(c, Bin::LShr, rm, ri);
            let bb = self.s.m.bin(c, Bin::LShr, bm, bi);
            let both = self.s.m.bin(c, Bin::And, rb, bb);
            let (one, z) = (k(self, 1), k(self, 0));
            let both = self.s.m.bin(c, Bin::And, both, one);
            let on = self.s.m.cmp(c, pred::NE, both, z);
            enable = Some(match enable {
                Some(e) => self.s.m.bin(c, Bin::And, e, on),
                None => on,
            });
        }
        Ok((r, enable))
    }

    fn valu_op(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let c = self.cur;
        let f = |l: &mut Self, k| l.fsrc(inst, k);
        let s = |l: &mut Self, k| l.src(inst, k);
        let float = |l: &mut Self, r: V| l.write_f(inst, r, guard);
        let int = |l: &mut Self, r: V| l.write_i(inst, r, guard);
        if let Some(r) = self.valu16(inst, name, guard) {
            return r;
        }
        if name == "v_pk_fmac_f16" {
            return self.packed_fmac(inst, guard);
        }
        if let Some(op) = name.strip_prefix("v_pk_") {
            return self.packed16(inst, op, guard);
        }
        match name {
            "v_frexp_exp_i16_f16" | "v_frexp_exp_i32_f32" | "v_frexp_exp_i32_f64"
            | "v_frexp_mant_f16" | "v_frexp_mant_f32" | "v_frexp_mant_f64" => {
                self.frexp(inst, guard)
            }
            // v_ldexp_f32 lowers below (exact while the result stays normal).
            "v_ldexp_f16" | "v_ldexp_f64" => {
                Err("ldexp requires the console floating-point rounding and denormal mode")
            }
            "v_mullit_f32" => Err("lighting multiply needs the console NaN, infinity and overflow rules"),
            "v_dot2_f32_f16" | "v_dot2c_f32_f16" => Err("half dot product rounding is unspecified by the emitter"),
            "v_fmamk_f16" | "v_fmaak_f16" => Err("half FMA requires a single rounding to f16"),
            n if n.starts_with("v_cvt_norm_") || n.starts_with("v_cvt_pknorm_") => {
                Err("normalized conversion needs the console rounding and NaN rules")
            }
            "v_mad_i32_i16" | "v_mad_u32_u16" => {
                let signed = !name.ends_with("u16");
                if v.omod != 0 || v.abs != 0 || v.neg != 0 || v.opsel & !3 != 0 {
                    return Err("modifiers on integer arithmetic");
                }
                let read = |l: &mut Self, i| -> Result<V, Why> {
                    let x = l.half_src(inst, i)?;
                    Ok(if signed { l.sign_extend(x, 16) } else { x })
                };
                let (a, b) = (read(self, 0)?, read(self, 1)?);
                let d = s(self, 2)?;
                let arithmetic = |l: &mut Self, a, b, d| {
                    let m = l.s.m.bin(c, Bin::Mul, a, b);
                    l.s.m.bin(c, Bin::Add, m, d)
                };
                let r = if v.clamp {
                    let (a, b) = (self.extend64(a, signed), self.extend64(b, signed));
                    let d = self.extend64(d, signed);
                    let r = arithmetic(self, a, b, d);
                    let (min, max) = if signed { (i32::MIN as i64, i32::MAX as i64) } else { (0, u32::MAX as i64) };
                    let r = self.clamp_int(r, 64, min, max, signed);
                    self.s.m.cast(c, Cast::Trunc, r, self.s.t.i32)
                } else {
                    arithmetic(self, a, b, d)
                };
                let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("a result without a VGPR") };
                self.write(inst, Reg::V(reg), r, guard);
                Ok(())
            }
            "v_cvt_pk_u16_u32" | "v_cvt_pk_i16_i32" | "v_sat_pk_u8_i16" => {
                // LLVM's AMDGPU clamp combines use the saturating V_CVT_PK
                // conversions. V_SAT_PK clamps each signed half to a byte.
                let a = s(self, 0)?;
                let r = if name == "v_sat_pk_u8_i16" {
                    let (k16, k8) = (self.s.m.i32(16), self.s.m.i32(8));
                    let hi = self.s.m.bin(c, Bin::AShr, a, k16);
                    let lo = self.sign_extend(a, 16);
                    let lo = self.clamp_int(lo, 32, 0, 255, true);
                    let hi = self.clamp_int(hi, 32, 0, 255, true);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k8);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                } else {
                    let b = s(self, 1)?;
                    let signed = name == "v_cvt_pk_i16_i32";
                    let (min, max) = if signed { (-32768, 32767) } else { (0, 65535) };
                    let (a, b) = (self.clamp_int(a, 32, min, max, signed), self.clamp_int(b, 32, min, max, signed));
                    self.pack_halves(a, b)
                };
                int(self, r)
            }
            "v_cvt_off_f32_i4" => {
                // RDNA 2 ISA, VOP1: a signed nibble in sixteenths of a pixel.
                let x = s(self, 0)?;
                let x = self.sign_extend(x, 4);
                let x = self.s.m.cast(c, Cast::SiToFp, x, self.s.t.f32);
                let step = self.s.m.f32(0.0625f32.to_bits());
                let x = self.s.m.bin(c, Bin::Mul, x, step);
                float(self, x)
            }
            "v_dot2_i32_i16" | "v_dot2_u32_u16" | "v_dot4_i32_i8" | "v_dot4c_i32_i8"
            | "v_dot4_u32_u8" | "v_dot8_u32_u4" => self.integer_dot(inst, guard),
            "v_mad_u64_u32" | "v_mad_i64_i32" => self.wide_mad(inst, guard),
            "v_qsad_pk_u16_u8" | "v_mqsad_pk_u16_u8" | "v_mqsad_u32_u8" => {
                if v.clamp || v.omod != 0 || v.abs != 0 || v.neg != 0 || v.opsel != 0 {
                    return Err("modifiers on a quad SAD");
                }
                let a = self.src64(inst, 0)?;
                let b = s(self, 1)?;
                let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("a quad SAD result") };
                let acc = if name == "v_mqsad_u32_u8" {
                    self.vgprs(inst, &v.src[2], 4)?
                } else {
                    let acc = self.src64(inst, 2)?;
                    let (lo, hi) = self.split64(acc);
                    vec![lo, hi]
                };
                let mut packed_lo = None;
                for i in 0..4u8 {
                    let ty = self.s.m.int(64);
                    let sh = self.s.m.int_const(ty, i64::from(i) * 8);
                    let window = self.s.m.bin(c, Bin::LShr, a, sh);
                    let window = self.s.m.cast(c, Cast::Trunc, window, self.s.t.i32);
                    let diff = self.sad(window, b, 8, name != "v_qsad_pk_u16_u8");
                    if acc.len() == 4 {
                        let r = self.s.m.bin(c, Bin::Add, diff, acc[usize::from(i)]);
                        self.write(inst, Reg::V(reg + i), r, guard);
                    } else {
                        let d = self.int_field(acc[usize::from(i / 2)], u32::from(i % 2) * 16, 16);
                        let r = self.s.m.bin(c, Bin::Add, diff, d);
                        if let Some(lo) = packed_lo.take() {
                            let r = self.pack_halves(lo, r);
                            self.write(inst, Reg::V(reg + i / 2), r, guard);
                        } else {
                            packed_lo = Some(r);
                        }
                    }
                }
                Ok(())
            }
            "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => {
                let n = s(self, 0)?;
                let x = self.src64(inst, 1)?;
                let i64t = self.s.m.int(64);
                let k63 = self.s.m.i32(63);
                let n = self.s.m.bin(c, Bin::And, n, k63);
                let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                let op = match name {
                    "v_lshlrev_b64" => Bin::Shl,
                    "v_lshrrev_b64" => Bin::LShr,
                    _ => Bin::AShr,
                };
                let r = self.s.m.bin(c, op, x, n);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 64-bit result");
                };
                let (lo, hi) = self.split64(r);
                self.write(inst, Reg::V(reg), lo, guard);
                self.write(inst, Reg::V(reg + 1), hi, guard);
                Ok(())
            }
            "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => {
                // Cube-map face coordinates (AMD's ISA): the major axis is z
                // when |z| >= |x| and |z| >= |y|, else y when |y| >= |x|,
                // else x.
                let (x, y, z) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let (ax, ay, az) = (self.unary(6, x), self.unary(6, y), self.unary(6, z));
                let zx = self.s.m.cmp(c, pred::OGE, az, ax);
                let zy = self.s.m.cmp(c, pred::OGE, az, ay);
                let zmaj = self.s.m.bin(c, Bin::And, zx, zy);
                let ymaj = self.s.m.cmp(c, pred::OGE, ay, ax);
                let zero = self.s.m.f32(0);
                let neg = |l: &mut Self, v: V| l.s.m.cmp(c, pred::OLT, v, zero);
                let minus = |l: &mut Self, v: V| {
                    let z = l.s.m.f32(0x8000_0000);
                    l.s.m.bin(c, Bin::Sub, z, v)
                };
                let pick = |l: &mut Self, zv: V, yv: V, xv: V| {
                    let t = l.s.m.select(c, ymaj, yv, xv);
                    l.s.m.select(c, zmaj, zv, t)
                };
                let r = match name {
                    "v_cubeid_f32" => {
                        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
                        let (nz, ny, nx) = (neg(self, z), neg(self, y), neg(self, x));
                        let (f5, f4, f3, f2, f1, f0) = (
                            k(self, 5.0),
                            k(self, 4.0),
                            k(self, 3.0),
                            k(self, 2.0),
                            k(self, 1.0),
                            k(self, 0.0),
                        );
                        let zf = self.s.m.select(c, nz, f5, f4);
                        let yf = self.s.m.select(c, ny, f3, f2);
                        let xf = self.s.m.select(c, nx, f1, f0);
                        pick(self, zf, yf, xf)
                    }
                    "v_cubesc_f32" => {
                        let (nz, nx) = (neg(self, z), neg(self, x));
                        let (mx, mz) = (minus(self, x), minus(self, z));
                        let zf = self.s.m.select(c, nz, mx, x);
                        let xf = self.s.m.select(c, nx, z, mz);
                        pick(self, zf, x, xf)
                    }
                    "v_cubetc_f32" => {
                        let ny = neg(self, y);
                        let (my, mz) = (minus(self, y), minus(self, z));
                        let yf = self.s.m.select(c, ny, mz, z);
                        pick(self, my, yf, my)
                    }
                    _ => {
                        let t = pick(self, z, y, x);
                        let two = self.s.m.f32(2.0f32.to_bits());
                        self.s.m.bin(c, Bin::Mul, t, two)
                    }
                };
                float(self, r)
            }
            "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32" => {
                // M0 indexes the source (movrels), the destination (movreld)
                // or both, from the encoded register up to the program's
                // VGPR count (the registers the instruction reads or writes).
                let m0 = self.i32_of(inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?)?;
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("an M0-relative move");
                };
                let indexed_src = name != "v_movreld_b32";
                let value = if indexed_src {
                    let mut r = self.s.m.i32(0);
                    for k in 0..=255u8.saturating_sub(sreg) {
                        let Some(x) = inst.read(Reg::V(sreg + k)) else {
                            break;
                        };
                        let x = self.i32_of(x)?;
                        let kk = self.s.m.i32(u32::from(k));
                        let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                        r = self.s.m.select(c, is, x, r);
                    }
                    r
                } else {
                    s(self, 0)?
                };
                if name == "v_movrels_b32" {
                    return int(self, value);
                }
                for k in 0..=255u8.saturating_sub(dreg) {
                    let Some(_) = inst.write(Reg::V(dreg + k)) else {
                        break;
                    };
                    let old = match inst.read(Reg::V(dreg + k)) {
                        Some(o) => self.i32_of(o)?,
                        None => self.s.m.i32(0),
                    };
                    let kk = self.s.m.i32(u32::from(k));
                    let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                    let x = self.s.m.select(c, is, value, old);
                    self.write_placed(inst, Reg::V(dreg + k), x, guard);
                }
                Ok(())
            }
            "v_swap_b32" => {
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("v_swap_b32");
                };
                let a = self.i32_of(inst.read(Reg::V(dreg)).ok_or("v_swap_b32")?)?;
                let b = self.i32_of(inst.read(Reg::V(sreg)).ok_or("v_swap_b32")?)?;
                self.write_placed(inst, Reg::V(dreg), b, guard);
                if dreg != sreg {
                    self.write_placed(inst, Reg::V(sreg), a, guard);
                }
                Ok(())
            }
            "v_ffbh_u32" | "v_ffbh_i32" => {
                // Counted from the top bit, -1 when none (DXIL's too).
                let a = s(self, 0)?;
                let op = if name == "v_ffbh_u32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                int(self, r)
            }
            "v_mov_b32" => {
                if v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 {
                    let a = f(self, 0)?;
                    return float(self, a);
                }
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32"
            | "v_max_f32" | "v_min_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let r = match name {
                    "v_add_f32" => self.s.m.bin(c, Bin::Add, a, bb),
                    "v_sub_f32" => self.s.m.bin(c, Bin::Sub, a, bb),
                    "v_subrev_f32" => self.s.m.bin(c, Bin::Sub, bb, a),
                    "v_mul_f32" => self.s.m.bin(c, Bin::Mul, a, bb),
                    "v_mul_legacy_f32" => self.legacy_mul(a, bb),
                    "v_max_f32" => self.max_min(true, a, bb),
                    _ => self.max_min(false, a, bb),
                };
                float(self, r)
            }
            "v_fma_f32" | "v_fmaak_f32" | "v_fmamk_f32" => {
                // shortcut: DXIL has no fused f32 multiply-add; FMad lets the driver fuse.
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = self.fmad(a, bb, cc);
                float(self, r)
            }
            "v_div_scale_f32" => {
                // RDNA2 ISA: src0 = value to scale, src1 = denominator, src2 = numerator.
                let (a, den, num) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let (ai, ni, di) = (self.as_int(a), self.as_int(num), self.as_int(den));
                let (k23, k255, kabs) = (
                    self.s.m.i32(23),
                    self.s.m.i32(0xFF),
                    self.s.m.i32(0x7FFF_FFFF),
                );
                let (zero, k23_exp, k232_exp, k96) = (
                    self.s.m.i32(0),
                    self.s.m.i32(23),
                    self.s.m.i32(232),
                    self.s.m.i32(96),
                );
                let exp_of = |l: &mut Self, x: V| {
                    let s = l.s.m.bin(c, Bin::LShr, x, k23);
                    l.s.m.bin(c, Bin::And, s, k255)
                };
                let (e1, e2) = (exp_of(self, ni), exp_of(self, di));
                let n_abs = self.s.m.bin(c, Bin::And, ni, kabs);
                let d_abs = self.s.m.bin(c, Bin::And, di, kabs);
                let z1 = self.s.m.cmp(c, pred::EQ, n_abs, zero);
                let z2 = self.s.m.cmp(c, pred::EQ, d_abs, zero);
                let any_zero = self.s.m.bin(c, Bin::Or, z1, z2);
                let special = {
                    let s1 = self.s.m.cmp(c, pred::EQ, e1, k255);
                    let s2 = self.s.m.cmp(c, pred::EQ, e2, k255);
                    let a12 = self.s.m.bin(c, Bin::Or, s1, s2);
                    self.s.m.bin(c, Bin::Or, a12, any_zero)
                };
                let is_den = self.s.m.cmp(c, pred::EQ, ai, di);
                let diff = self.s.m.bin(c, Bin::Sub, e2, e1);
                let den_tiny = self.s.m.cmp(c, pred::IULE, e2, k23_exp);
                let den_huge = self.s.m.cmp(c, pred::IUGE, e2, k232_exp);
                let num_tiny_exp = self.s.m.cmp(c, pred::IULE, e1, k23_exp);
                let num_huge = self.s.m.cmp(c, pred::IUGE, e1, k232_exp);
                let num_gap = self.s.m.cmp(c, pred::SGE, diff, k96);
                let not_den_huge = {
                    let t = self.s.m.bool(true);
                    self.s.m.bin(c, Bin::Xor, den_huge, t)
                };
                let num_gap = self.s.m.bin(c, Bin::And, num_gap, not_den_huge);
                let num_tiny = self.s.m.bin(c, Bin::Or, num_tiny_exp, num_gap);
                let up_raw = self.s.m.select(c, is_den, den_tiny, num_tiny);
                let dn_raw = self.s.m.select(c, is_den, den_huge, num_huge);
                let not_special = {
                    let t = self.s.m.bool(true);
                    self.s.m.bin(c, Bin::Xor, special, t)
                };
                let up = self.s.m.bin(c, Bin::And, up_raw, not_special);
                let dn = self.s.m.bin(c, Bin::And, dn_raw, not_special);
                let uncancelled = {
                    let x_tiny = self.s.m.bin(c, Bin::Xor, den_tiny, num_tiny);
                    let x_huge = self.s.m.bin(c, Bin::Xor, den_huge, num_huge);
                    self.s.m.bin(c, Bin::Or, x_tiny, x_huge)
                };
                let flag_raw = self.s.m.select(c, is_den, uncancelled, up_raw);
                let flag = self.s.m.bin(c, Bin::And, flag_raw, not_special);
                let k_up = self.s.m.f32(0x5F80_0000);
                let k_dn = self.s.m.f32(0x1F80_0000);
                let a_up = self.s.m.bin(c, Bin::Mul, a, k_up);
                let a_dn = self.s.m.bin(c, Bin::Mul, a, k_dn);
                let r = self.s.m.select(c, dn, a_dn, a);
                let r = self.s.m.select(c, up, a_up, r);
                let qnan = self.s.m.f32(0x7FC0_0000);
                let r = self.s.m.select(c, any_zero, qnan, r);
                float(self, r)?;
                if let Some(Operand::Scalar { code, .. }) = v.carry
                    && let Some(w) = inst.write(Reg::S(code))
                {
                    self.lane_bits.insert(w, flag);
                }
                for &(_, w) in &inst.writes {
                    self.vals
                        .entry(w)
                        .or_insert(Val::Missing("a lane mask read as data"));
                }
                Ok(())
            }
            "v_div_fmas_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let m = inst
                    .read(Reg::S(crate::ir::VCC))
                    .and_then(|v| self.x.st.masks.get(&v).copied())
                    .ok_or("v_div_fmas_f32 without VCC")?;
                let vcc = self.boolean(m, c)?;
                let tmp = self.fmad(a, bb, cc);
                let ci = self.as_int(cc);
                let (k23, k255, k128) = (self.s.m.i32(23), self.s.m.i32(0xFF), self.s.m.i32(128));
                let sh = self.s.m.bin(c, Bin::LShr, ci, k23);
                let exp = self.s.m.bin(c, Bin::And, sh, k255);
                let exp_ge = self.s.m.cmp(c, pred::IUGE, exp, k128);
                let (k_up, k_dn) = (self.s.m.f32(0x5F80_0000), self.s.m.f32(0x1F80_0000));
                let scale = self.s.m.select(c, exp_ge, k_up, k_dn);
                let scaled = self.s.m.bin(c, Bin::Mul, tmp, scale);
                let r = self.s.m.select(c, vcc, scaled, tmp);
                float(self, r)
            }
            "v_div_fixup_f32" => {
                let (q, den, num) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let (qi, di, ni) = (self.as_int(q), self.as_int(den), self.as_int(num));
                let (ksign, kabs, kinf) = (
                    self.s.m.i32(0x8000_0000),
                    self.s.m.i32(0x7FFF_FFFF),
                    self.s.m.i32(0x7F80_0000),
                );
                let (kqnan_bit, kneg_qnan, zero) = (
                    self.s.m.i32(0x0040_0000),
                    self.s.m.i32(0xFFC0_0000),
                    self.s.m.i32(0),
                );
                let sign = {
                    let x = self.s.m.bin(c, Bin::Xor, ni, di);
                    self.s.m.bin(c, Bin::And, x, ksign)
                };
                let q_abs = self.s.m.bin(c, Bin::And, qi, kabs);
                let d_abs = self.s.m.bin(c, Bin::And, di, kabs);
                let n_abs = self.s.m.bin(c, Bin::And, ni, kabs);
                let q_nan = self.s.m.cmp(c, pred::IUGT, q_abs, kinf);
                let d_nan = self.s.m.cmp(c, pred::IUGT, d_abs, kinf);
                let n_nan = self.s.m.cmp(c, pred::IUGT, n_abs, kinf);
                let d_inf = self.s.m.cmp(c, pred::EQ, d_abs, kinf);
                let n_inf = self.s.m.cmp(c, pred::EQ, n_abs, kinf);
                let d_zero = self.s.m.cmp(c, pred::EQ, d_abs, zero);
                let n_zero = self.s.m.cmp(c, pred::EQ, n_abs, zero);
                let both_inf = self.s.m.bin(c, Bin::And, n_inf, d_inf);
                let both_zero = self.s.m.bin(c, Bin::And, n_zero, d_zero);
                let invalid = self.s.m.bin(c, Bin::Or, both_inf, both_zero);
                let make_inf = {
                    let t = self.s.m.bin(c, Bin::Or, n_inf, d_zero);
                    self.s.m.bin(c, Bin::Or, t, q_nan)
                };
                let make_zero = self.s.m.bin(c, Bin::Or, d_inf, n_zero);
                let signed_q = self.s.m.bin(c, Bin::Or, sign, q_abs);
                let signed_inf = self.s.m.bin(c, Bin::Or, sign, kinf);
                let n_qnan = self.s.m.bin(c, Bin::Or, ni, kqnan_bit);
                let d_qnan = self.s.m.bin(c, Bin::Or, di, kqnan_bit);
                let r = self.s.m.select(c, make_zero, sign, signed_q);
                let r = self.s.m.select(c, make_inf, signed_inf, r);
                let r = self.s.m.select(c, invalid, kneg_qnan, r);
                let r = self.s.m.select(c, d_nan, d_qnan, r);
                let r = self.s.m.select(c, n_nan, n_qnan, r);
                let rf = self.as_float(r);
                float(self, rf)
            }
            "v_fmac_f32" | "v_mac_f32" | "v_fmac_legacy_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let acc = inst.read(Reg::V(reg)).ok_or("an accumulator")?;
                let acc = match self.floats.get(&acc) {
                    Some(&x) => x,
                    None => {
                        let x = self.i32_of(acc)?;
                        self.as_float(x)
                    }
                };
                let r = match name {
                    "v_fmac_f32" => self.fmad(a, bb, acc),
                    "v_mac_f32" => {
                        let m = self.s.m.bin(c, Bin::Mul, a, bb);
                        self.s.m.bin(c, Bin::Add, m, acc)
                    }
                    _ => self.fma_legacy(a, bb, acc),
                };
                float(self, r)
            }
            "v_fma_legacy_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = self.fma_legacy(a, bb, cc);
                float(self, r)
            }
            "v_mad_f32" | "v_madak_f32" | "v_madmk_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let m = self.s.m.bin(c, Bin::Mul, a, bb);
                let r = self.s.m.bin(c, Bin::Add, m, cc);
                float(self, r)
            }
            "v_max3_f32" | "v_min3_f32" | "v_med3_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = match name {
                    "v_max3_f32" => {
                        let t = self.max_min(true, a, bb);
                        self.max_min(true, t, cc)
                    }
                    "v_min3_f32" => {
                        let t = self.max_min(false, a, bb);
                        self.max_min(false, t, cc)
                    }
                    _ => self.med3(a, bb, cc),
                };
                float(self, r)
            }
            "v_rcp_f32" | "v_rcp_iflag_f32" => {
                // shortcut: the hardware reciprocal is approximate (1 ULP); this divides.
                let a = f(self, 0)?;
                let one = self.s.m.f32(1.0f32.to_bits());
                let r = self.s.m.bin(c, Bin::SDiv, one, a);
                float(self, r)
            }
            "v_rsq_f32" | "v_sqrt_f32" | "v_log_f32" | "v_exp_f32" | "v_fract_f32"
            | "v_trunc_f32" | "v_floor_f32" | "v_ceil_f32" | "v_rndne_f32" | "v_sin_f32"
            | "v_cos_f32" => {
                let a = f(self, 0)?;
                let op = match name {
                    "v_rsq_f32" => 25,
                    "v_sqrt_f32" => 24,
                    "v_log_f32" => 23,
                    "v_exp_f32" => 21,
                    "v_fract_f32" => 22,
                    "v_trunc_f32" => 29,
                    "v_floor_f32" => 27,
                    "v_ceil_f32" => 28,
                    "v_rndne_f32" => 26,
                    "v_sin_f32" => 13,
                    _ => 12,
                };
                let r = match op {
                    12 | 13 => self.trig_turns(a, op == 12),
                    22 => self.fract(a, 1.0, f32::from_bits(0x3F7F_FFFF)),
                    _ => self.unary(op, a),
                };
                float(self, r)
            }
            "v_cvt_i32_f32" | "v_cvt_u32_f32" | "v_cvt_flr_i32_f32" | "v_cvt_rpi_i32_f32" => {
                let mut a = f(self, 0)?;
                if name == "v_cvt_rpi_i32_f32" {
                    let h = self.s.m.f32(0.5f32.to_bits());
                    a = self.s.m.bin(c, Bin::Add, a, h);
                }
                if name != "v_cvt_i32_f32" && name != "v_cvt_u32_f32" {
                    a = self.unary(27, a);
                }
                // D3D's conversions saturate and give 0 for NaN, as the
                // hardware does (checked on the RX 6700 XT and WARP).
                let op = if name == "v_cvt_u32_f32" {
                    Cast::FpToUi
                } else {
                    Cast::FpToSi
                };
                let r = self.s.m.cast(c, op, a, self.s.t.i32);
                int(self, r)
            }
            "v_cvt_f32_i32" | "v_cvt_f32_u32" | "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1"
            | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => {
                let mut a = s(self, 0)?;
                if let Some(n) = name.strip_prefix("v_cvt_f32_ubyte") {
                    let sh = self.s.m.i32(8 * u32::from(n.as_bytes()[0] - b'0'));
                    let ff = self.s.m.i32(0xff);
                    a = self.s.m.bin(c, Bin::LShr, a, sh);
                    a = self.s.m.bin(c, Bin::And, a, ff);
                }
                let op = if name == "v_cvt_f32_i32" {
                    Cast::SiToFp
                } else {
                    Cast::UiToFp
                };
                let r = self.s.m.cast(c, op, a, self.s.t.f32);
                float(self, r)
            }
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                // Each source is f32, or with op_sel_hi the low (op_sel: high)
                // f16 half; neg_hi is the absolute value here.
                let mut xs = Vec::new();
                for i in 0..3 {
                    let op = v.src.get(i).ok_or("a missing operand")?;
                    let mut x = if v.opsel_hi >> i & 1 != 0 {
                        let x = self.f16_operand(inst, op)?;
                        let h = if v.opsel >> i & 1 != 0 {
                            let k16 = self.s.m.i32(16);
                            self.s.m.bin(c, Bin::LShr, x, k16)
                        } else {
                            x
                        };
                        self.s
                            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h])
                    } else {
                        let x = self.operand(inst, op)?;
                        self.as_float(x)
                    };
                    if v.neg_hi >> i & 1 != 0 {
                        x = self.unary(6, x);
                    }
                    if v.neg >> i & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        x = self.s.m.bin(c, Bin::Sub, z, x);
                    }
                    xs.push(x);
                }
                let mut r = self.fmad(xs[0], xs[1], xs[2]);
                if name == "v_fma_mix_f32" {
                    return float(self, r);
                }
                if v.clamp {
                    r = self.unary(7, r);
                }
                // The f16 result goes to one half; the other half stays.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a mixed-precision result");
                };
                let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
                    Some(Ok(o)) => o,
                    _ => self.s.m.i32(0),
                };
                let h = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[r]);
                let (keep, h) = if name == "v_fma_mixhi_f16" {
                    let k16 = self.s.m.i32(16);
                    (0xFFFF, self.s.m.bin(c, Bin::Shl, h, k16))
                } else {
                    (0xFFFF_0000, h)
                };
                let keep = self.s.m.i32(keep);
                let old = self.s.m.bin(c, Bin::And, old, keep);
                let x = self.s.m.bin(c, Bin::Or, old, h);
                self.write(inst, Reg::V(reg), x, guard);
                Ok(())
            }
            "v_cvt_pkrtz_f16_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let lo = self.f16_toward_zero(a);
                let hi = self.f16_toward_zero(bb);
                let sh = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, hi, sh);
                let r = self.s.m.bin(c, Bin::Or, lo, hi);
                int(self, r)
            }
            "v_ldexp_f32" => {
                // src0 * 2^src1 as three factors 2^e, each exponent in the
                // normal range, so the products are exact until the result
                // leaves it. Past |src1| = 375 every finite nonzero src0 has
                // over- or underflowed, so src1 is clamped there; then
                // e1 = floor(src1 * 21846 / 65536) (about a third) and the
                // rest in two halves.
                // shortcut: a result that turns denormal before the last
                // factor is rounded twice where denormals are kept.
                let (a, e) = (f(self, 0)?, s(self, 1)?);
                let (lo, hi) = (self.s.m.i32((-375i32).cast_unsigned()), self.s.m.i32(375));
                let below = self.s.m.cmp(c, pred::SLT, e, lo);
                let e = self.s.m.select(c, below, lo, e);
                let above = self.s.m.cmp(c, pred::SGT, e, hi);
                let e = self.s.m.select(c, above, hi, e);
                let (third, k16, one) = (self.s.m.i32(21846), self.s.m.i32(16), self.s.m.i32(1));
                let t = self.s.m.bin(c, Bin::Mul, e, third);
                let e1 = self.s.m.bin(c, Bin::AShr, t, k16);
                let rest = self.s.m.bin(c, Bin::Sub, e, e1);
                let e2 = self.s.m.bin(c, Bin::AShr, rest, one);
                let e3 = self.s.m.bin(c, Bin::Sub, rest, e2);
                let mut r = a;
                for x in [e1, e2, e3] {
                    let (bias, k23) = (self.s.m.i32(127), self.s.m.i32(23));
                    let x = self.s.m.bin(c, Bin::Add, x, bias);
                    let x = self.s.m.bin(c, Bin::Shl, x, k23);
                    let p = self.as_float(x);
                    r = self.s.m.bin(c, Bin::Mul, r, p);
                }
                float(self, r)
            }
            "v_cvt_f16_f32" if v.opsel == 0 => {
                // Rounded to nearest even (the MODE default) into the low half.
                let a = f(self, 0)?;
                let r = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[a]);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 16-bit result outside a VGPR");
                };
                let r = self.low_half(inst, reg, r);
                int(self, r)
            }
            "v_pack_b32_f16" => {
                // Two f16 sources (op_sel: their high halves; abs and neg on
                // the sign bit) into the low and high half.
                let mut halves = Vec::new();
                for k in 0..2 {
                    let op = v.src.get(k).ok_or("a missing operand")?;
                    let mut x = self.f16_operand(inst, op)?;
                    if v.opsel >> k & 1 != 0 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let mask = if v.abs >> k & 1 != 0 { 0x7FFF } else { 0xFFFF };
                    let mask = self.s.m.i32(mask);
                    x = self.s.m.bin(c, Bin::And, x, mask);
                    if v.neg >> k & 1 != 0 {
                        let sign = self.s.m.i32(0x8000);
                        x = self.s.m.bin(c, Bin::Xor, x, sign);
                    }
                    halves.push(x);
                }
                let k16 = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
                let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
                int(self, r)
            }
            "v_cvt_f32_f16" if v.opsel == 0 => {
                let a = s(self, 0)?;
                let r = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[a]);
                float(self, r)
            }
            "v_cndmask_b32" => {
                let cond = self.select_condition(inst, v.src.get(2))?;
                if v.abs != 0 || v.neg != 0 {
                    let (a, bb) = (f(self, 0)?, f(self, 1)?);
                    let r = self.s.m.select(c, cond, bb, a);
                    return float(self, r);
                }
                let (a, bb) = (s(self, 0)?, s(self, 1)?);
                let r = self.s.m.select(c, cond, bb, a);
                int(self, r)
            }
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32"
            | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => {
                // The carry (borrow) out is a lane mask: this lane's bit.
                let (mut a, mut bb) = (s(self, 0)?, s(self, 1)?);
                if name.contains("subrev") {
                    std::mem::swap(&mut a, &mut bb);
                }
                let cin = if name.ends_with("_ci_u32") {
                    let m = match v.src.get(2) {
                        Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)),
                        _ => None,
                    };
                    let m = m
                        .and_then(|m| self.x.st.masks.get(&m).copied())
                        .ok_or("a carry-in from an untracked mask")?;
                    Some(self.boolean(m, c)?)
                } else {
                    None
                };
                let (r, carry) = self.add_sub_carry(c, name.starts_with("v_add"), a, bb, cin);
                int(self, r)?;
                if let Some(Operand::Scalar { code, .. }) = v.carry
                    && let Some(w) = inst.write(Reg::S(code))
                {
                    self.lane_bits.insert(w, carry);
                }
                for &(_, w) in &inst.writes {
                    self.vals
                        .entry(w)
                        .or_insert(Val::Missing("a lane mask read as data"));
                }
                Ok(())
            }
            // Each lane reads a lane of its row of 16, or of the other row of
            // its half of 32, picked by its nibble of {src2, src1}.
            "v_permlane16_b32" | "v_permlanex16_b32" => {
                let (a, s1, s2) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                self.cross_lane(false)?;
                if self.split || self.console_cf {
                    return Err("a lane permutation in a program split in halves");
                }
                let l = self.lane_index();
                let k = |l: &mut Self, x: u32| l.s.m.i32(x);
                let (k7, k8, k2, k15, top, zero) =
                    (k(self, 7), k(self, 8), k(self, 2), k(self, 15), k(self, !15), k(self, 0));
                let n = self.s.m.bin(c, Bin::And, l, k7);
                let n = self.s.m.bin(c, Bin::Shl, n, k2);
                let half = self.s.m.bin(c, Bin::And, l, k8);
                let high = self.s.m.cmp(c, pred::NE, half, zero);
                let word = self.s.m.select(c, high, s2, s1);
                let pick = self.s.m.bin(c, Bin::LShr, word, n);
                let pick = self.s.m.bin(c, Bin::And, pick, k15);
                let mut row = self.s.m.bin(c, Bin::And, l, top);
                if name == "v_permlanex16_b32" {
                    let k16 = self.s.m.i32(16);
                    row = self.s.m.bin(c, Bin::Xor, row, k16);
                }
                let lane = self.s.m.bin(c, Bin::Or, row, pick);
                let r = self.read_lane(a, lane);
                int(self, r)
            }
            "v_readlane_b32"
                if matches!(v.src.first(), Some(&Operand::Vector { reg, .. })
                    if inst.read(Reg::V(reg)).is_some_and(|r| self.one_lane.contains(&r))) =>
            {
                // Made from that lane's value already.
                let r = s(self, 0)?;
                int(self, r)
            }
            "v_readlane_b32" => {
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                // A computed lane in a pixel or vertex program is a host lane
                // (from a mask or `v_mbcnt`); a constant one is the console's.
                if self.mask_data == MaskData::Host && !matches!(lane, V::Const(_)) {
                    self.s.flags |= flags::WAVE_OPS;
                } else {
                    self.cross_lane(false)?;
                }
                let r = if self.console_cf {
                    self.console_read_lane(a, lane)
                } else {
                    self.read_lane(a, lane)
                };
                int(self, r)
            }
            "v_writelane_b32" => {
                // The lane given takes the scalar; the others keep their value.
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a lane write");
                };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.i32_of(o)?,
                    None => self.s.m.i32(0),
                };
                self.cross_lane(false)?;
                let me = if self.console_cf {
                    self.console_lane().0
                } else {
                    self.lane_index()
                };
                let is = self.s.m.cmp(c, pred::EQ, me, lane);
                let r = self.s.m.select(c, is, a, old);
                int(self, r)
            }
            "v_readfirstlane_b32"
                if inst
                    .writes
                    .first()
                    .is_some_and(|w| self.x.res.waterfall.reads.contains(&w.1)) =>
            {
                // A waterfall loop's read of a descriptor dword: the lane's own.
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_readfirstlane_b32" if self.console_cf => {
                // The console's first lane in EXEC (lane 0 when none).
                let a = s(self, 0)?;
                let word = |l: &mut Self, r: u8| -> Result<V, Why> {
                    match l.get(inst.read(Reg::S(r)).ok_or("EXEC")?)? {
                        Val::I32(x) => Ok(x),
                        _ => Err(LANE_MASK),
                    }
                };
                let lo = word(self, EXEC)?;
                let hi = if self.x.native.wave64 {
                    word(self, EXEC + 1)?
                } else {
                    self.s.m.i32(0)
                };
                let i32t = self.s.t.i32;
                let flo = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[lo]);
                let fhi = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[hi]);
                let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
                let fhi = self.s.m.bin(c, Bin::Add, fhi, k32);
                let none_hi = self.s.m.cmp(c, pred::EQ, hi, z);
                let fhi = self.s.m.select(c, none_hi, z, fhi);
                let some_lo = self.s.m.cmp(c, pred::NE, lo, z);
                let first = self.s.m.select(c, some_lo, flo, fhi);
                let r = self.console_read_lane(a, first);
                int(self, r)
            }
            "v_readfirstlane_b32" => {
                let a = s(self, 0)?;
                self.s.flags |= flags::WAVE_OPS;
                let r = self.s.op(
                    c,
                    "waveReadLaneFirst.i32",
                    Attr::NoUnwind,
                    self.s.t.i32,
                    118,
                    &[a],
                );
                let r = self.helper_reads_own(a, r);
                int(self, r)
            }
            "v_add_f64" | "v_mul_f64" | "v_fma_f64" => {
                if v.omod != 0 || v.clamp {
                    return Err("f64 output modifiers");
                }
                let a = self.src_f64(inst, 0)?;
                let b = self.src_f64(inst, 1)?;
                let r = match name {
                    "v_add_f64" => self.s.m.bin(c, Bin::Add, a, b),
                    "v_mul_f64" => self.s.m.bin(c, Bin::Mul, a, b),
                    _ => {
                        let d = self.src_f64(inst, 2)?;
                        self.s.flags |= flags::DOUBLE_EXT;
                        let f64t = self.s.m.double();
                        self.s
                            .op(c, "fma.f64", Attr::ReadNone, f64t, 47, &[a, b, d])
                    }
                };
                self.write_f64(inst, r, guard)
            }
            "v_cvt_f32_f64" => {
                let a = self.src_f64(inst, 0)?;
                let r = self.s.m.cast(c, Cast::FpTrunc, a, self.s.t.f32);
                float(self, r)
            }
            "v_cvt_i32_f64" | "v_cvt_u32_f64" => {
                if v.omod != 0 || v.clamp {
                    return Err("modifiers on integer conversion");
                }
                let a = self.src_f64(inst, 0)?;
                let cast = if name == "v_cvt_i32_f64" {
                    Cast::FpToSi
                } else {
                    Cast::FpToUi
                };
                let r = self.s.m.cast(c, cast, a, self.s.t.i32);
                int(self, r)
            }
            "v_cvt_f64_f32" | "v_cvt_f64_i32" | "v_cvt_f64_u32" => {
                if v.omod != 0 || v.clamp {
                    return Err("f64 output modifiers");
                }
                self.s.flags |= flags::DOUBLES;
                let f64t = self.s.m.double();
                let r = match name {
                    "v_cvt_f64_f32" => {
                        let a = f(self, 0)?;
                        self.s.m.cast(c, Cast::FpExt, a, f64t)
                    }
                    "v_cvt_f64_i32" => {
                        let a = s(self, 0)?;
                        self.s.m.cast(c, Cast::SiToFp, a, f64t)
                    }
                    _ => {
                        let a = s(self, 0)?;
                        self.s.m.cast(c, Cast::UiToFp, a, f64t)
                    }
                };
                self.write_f64(inst, r, guard)
            }
            n if n.contains("_f64") => Err("floating-point arithmetic in f64 is not implemented by the DXIL emitter"),
            _ => {
                let r = self.integer(inst, name)?;
                int(self, r)
            }
        }
    }

    /// A 64-bit floating-point source, constructed from its bit pattern via `dx.op.makeDouble`.
    fn src_f64(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let bits = self.fp_bits(inst, k, 64)?;
        let (lo, hi) = self.split64(bits);
        self.s.flags |= flags::DOUBLES;
        let f64t = self.s.m.double();
        Ok(self
            .s
            .op(self.cur, "makeDouble.f64", Attr::ReadNone, f64t, 101, &[lo, hi]))
    }

    /// Writes a 64-bit floating-point result into a VGPR pair via `dx.op.splitDouble`.
    fn write_f64(&mut self, inst: &Inst, val: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("an f64 destination");
        };
        self.s.flags |= flags::DOUBLES;
        let i32t = self.s.t.i32;
        let pair = self.s.m.named_struct("dx.types.splitdouble", vec![i32t, i32t]);
        let split = self
            .s
            .op(self.cur, "splitDouble.f64", Attr::ReadNone, pair, 102, &[val]);
        let lo = self.s.m.extract(self.cur, split, 0);
        let hi = self.s.m.extract(self.cur, split, 1);
        self.write(inst, Reg::V(reg), lo, guard);
        self.write(inst, Reg::V(reg + 1), hi, guard);
        Ok(())
    }

    /// A 64-bit source (a register pair; inline constants sign-extend).
    fn src64(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit operand modifiers");
        }
        self.raw_src64(inst, k, false)
    }

    fn raw_src64(&mut self, inst: &Inst, k: usize, float: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else { unreachable!() };
        if v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit SDWA or DPP operands");
        }
        let c = self.cur;
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let pair = |l: &mut Self, r: Reg, r1: Reg| -> Result<V, Why> {
            let lo = l.i32_of(inst.read(r).ok_or("an unread operand")?)?;
            let hi = l.i32_of(inst.read(r1).ok_or("an unread operand")?)?;
            Ok(l.join64(lo, hi))
        };
        match op {
            Operand::Vector { reg, .. } => pair(self, Reg::V(reg), Reg::V(reg + 1)),
            Operand::Scalar { code, .. } if code != NULL => {
                pair(self, Reg::S(code), Reg::S(code + 1))
            }
            Operand::Const(k @ 240..=248) if float => {
                let f = if k == 248 {
                    0.5 * std::f64::consts::FRAC_1_PI
                } else {
                    let b = crate::resource::inline(k).ok_or("a special operand")?;
                    f64::from(f32::from_bits(b))
                };
                Ok(self.s.m.int_const(i64t, f.to_bits() as i64))
            }
            Operand::Literal(x) if float => Ok(self.s.m.int_const(i64t, i64::from(x) << 32)),
            Operand::Literal(x) if inst.name() == "v_mad_i64_i32" => {
                Ok(self.s.m.int_const(i64t, i64::from(x as i32)))
            }
            Operand::Const(k @ 128..=208) => {
                let x = crate::resource::inline(k).ok_or("a special operand")?;
                Ok(self.s.m.int_const(i64t, i64::from(x as i32)))
            }
            _ => {
                // Literals and the null register: zero-extended.
                let x = self.operand(inst, &op)?;
                Ok(self.s.m.cast(c, Cast::ZExt, x, i64t))
            }
        }
    }

    /// IEEE source bits, with sign modifiers applied as bit operations so
    /// that a signalling NaN, a denormal or a negative zero is preserved.
    fn fp_bits(&mut self, inst: &Inst, k: usize, bits: u32) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else { unreachable!() };
        let mut x = match bits {
            16 => self.half_src(inst, k)?,
            32 => match (self.src0, k) {
                (Some(x), 0) => x,
                _ => {
                    let op = v.src.get(k).ok_or("a missing operand")?;
                    let x = self.operand(inst, op)?;
                    match Self::sdwa_src(inst, k) {
                        Some((sel, sext)) => self.select_bits(x, sel, sext),
                        None => x,
                    }
                }
            },
            _ => self.raw_src64(inst, k, true)?,
        };
        let ty = self.s.m.int(bits.max(32));
        let sign = 1u64 << (bits - 1);
        if v.abs >> k & 1 != 0 {
            let mask = self.s.m.int_const(ty, (sign - 1) as i64);
            x = self.s.m.bin(self.cur, Bin::And, x, mask);
        }
        if v.neg >> k & 1 != 0 {
            let sign = self.s.m.int_const(ty, sign as i64);
            x = self.s.m.bin(self.cur, Bin::Xor, x, sign);
        }
        Ok(x)
    }

    /// Decompose the IEEE encoding directly. AMD's RDNA 2 ISA specifies
    /// zero exponent for infinities and NaNs and passes their mantissa
    /// through. LLVM's AMDGPU frexp fold also normalizes subnormal inputs.
    fn frexp(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else { unreachable!() };
        if v.clamp || v.omod != 0 {
            return Err("floating output modifiers on frexp");
        }
        let name = inst.name();
        let (bits, fraction, bias) = if name.ends_with("f64") { (64, 52, 1023) }
            else if name.ends_with("f32") { (32, 23, 127) } else { (16, 10, 15) };
        let c = self.cur;
        let ty = self.s.m.int(bits.max(32));
        let x = self.fp_bits(inst, 0, bits)?;
        let k = |l: &mut Self, n| l.s.m.int_const(ty, n);
        let (z, frac, shift, emax, abs) = (k(self, 0), k(self, (1 << fraction) - 1),
            k(self, fraction), k(self, 2 * bias + 1), k(self, ((1u64 << (bits - 1)) - 1) as i64));
        let mant = self.s.m.bin(c, Bin::And, x, frac);
        let exp = self.s.m.bin(c, Bin::LShr, x, shift);
        let exp = self.s.m.bin(c, Bin::And, exp, emax);
        let subnormal = self.s.m.cmp(c, pred::EQ, exp, z);
        let abs = self.s.m.bin(c, Bin::And, x, abs);
        let nonzero = self.s.m.cmp(c, pred::NE, abs, z);
        let finite = self.s.m.cmp(c, pred::NE, exp, emax);
        let compute = self.s.m.bin(c, Bin::And, nonzero, finite);
        // Locate the highest set fraction bit with DXIL's i32 first-bit-high.
        let fb = |l: &mut Self, a| l.s.op(c, "unaryBits.i32", Attr::ReadNone, l.s.t.i32, 33, &[a]);
        let leading = if bits == 64 {
            let (lo, hi) = self.split64(mant);
            let (a, b) = (fb(self, lo), fb(self, hi));
            let (size, zero) = (self.s.m.i32(32), self.s.m.i32(0));
            let a = self.s.m.bin(c, Bin::Add, a, size);
            let low = self.s.m.cmp(c, pred::EQ, hi, zero);
            self.s.m.select(c, low, a, b)
        } else {
            fb(self, mant)
        };
        // A zero mantissa can produce -1. Mask the normalization shift
        // before emitting even the unselected path, so it is always defined.
        let offset = self.s.m.i32(bits.max(32) - 1 - fraction as u32);
        let left = self.s.m.bin(c, Bin::Sub, leading, offset);
        let limit = self.s.m.i32(bits.max(32) - 1);
        let left = self.s.m.bin(c, Bin::And, left, limit);
        let left = if bits == 64 { self.extend64(left, false) } else { left };
        let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("a frexp result") };
        if name.starts_with("v_frexp_exp_") {
            let one = k(self, 1);
            let e_sub = self.s.m.bin(c, Bin::Sub, one, left);
            let e = self.s.m.select(c, subnormal, e_sub, exp);
            let bias = k(self, bias - 1);
            let e = self.s.m.bin(c, Bin::Sub, e, bias);
            let e = self.s.m.select(c, compute, e, z);
            if bits == 16 {
                self.write_half(inst, e, guard)
            } else {
                let e = if bits == 64 { self.s.m.cast(c, Cast::Trunc, e, self.s.t.i32) } else { e };
                self.write_i(inst, e, guard)
            }
        } else {
            let norm = self.s.m.bin(c, Bin::Shl, mant, left);
            let mant = self.s.m.select(c, subnormal, norm, mant);
            let mant = self.s.m.bin(c, Bin::And, mant, frac);
            let sign = k(self, (1u64 << (bits - 1)) as i64);
            let sign = self.s.m.bin(c, Bin::And, x, sign);
            let half = k(self, (bias - 1) << fraction);
            let value = self.s.m.bin(c, Bin::Or, sign, half);
            let value = self.s.m.bin(c, Bin::Or, value, mant);
            let value = self.s.m.select(c, compute, value, x);
            match bits {
                16 => self.write_half(inst, value, guard),
                32 => self.write_i(inst, value, guard),
                _ => {
                    let (lo, hi) = self.split64(value);
                    self.write(inst, Reg::V(reg), lo, guard);
                    self.write(inst, Reg::V(reg + 1), hi, guard);
                    Ok(())
                }
            }
        }
    }

    /// Source `k` of a 16-bit operation in the low 16 bits (the rest 0):
    /// op_sel (or an SDWA word select) picks the high half; inline float
    /// constants are f16 values here.
    fn half_src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let mut high = v.opsel >> k & 1 != 0;
        if let Some(sd) = v.sdwa {
            match sd.src_sel.get(k).copied().unwrap_or(6) {
                4 => {}
                5 => high = true,
                6 => {}
                _ => return Err("a byte select on a 16-bit operation"),
            }
        }
        let x = match (op, self.src0) {
            (_, Some(x)) if k == 0 => x,
            (Operand::Const(c @ 240..=248), _) => self.s.m.i32(inline_f16(c)),
            _ => self.operand(inst, &op)?,
        };
        let c = self.cur;
        let x = if high {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// An f16 source as an f32 (exact), with its abs/neg modifiers.
    fn half_fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let h = self.half_src(inst, k)?;
        let f32t = self.s.t.f32;
        let mut f = self
            .s
            .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    /// Writes a 16-bit result (low bits of `x`) into the destination's low
    /// half, or its high half with op_sel bit 3 (or an SDWA WORD_1 select);
    /// the other half is kept (GFX10).
    fn write_half(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a 16-bit result outside a VGPR");
        };
        let mut high = v.opsel >> 3 & 1 != 0;
        // SDWA: the word written, the other one kept (preserve) or zero
        // (pad; a sign extension above the high word adds nothing).
        let mut keep = true;
        if let Some(sd) = v.sdwa {
            match (sd.dst_sel, sd.dst_unused) {
                (4, 2) => {}
                (5, 2) => high = true,
                (4 | 6, _) => keep = false,
                (5, _) => (high, keep) = (true, false),
                _ => return Err("an SDWA destination of a 16-bit operation"),
            }
        }
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) if keep => o,
            _ => self.s.m.i32(0),
        };
        let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
        let x = self.s.m.bin(c, Bin::And, x, low);
        let r = if high {
            let o = self.s.m.bin(c, Bin::And, old, low);
            let x = self.s.m.bin(c, Bin::Shl, x, k16);
            self.s.m.bin(c, Bin::Or, o, x)
        } else {
            let keep = self.s.m.i32(0xFFFF_0000);
            let o = self.s.m.bin(c, Bin::And, old, keep);
            self.s.m.bin(c, Bin::Or, o, x)
        };
        self.write_placed(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    /// 16-bit operations, computed in f32 or i32: f16 sums, differences and
    /// products are exact there and round once to f16 (f32 has the 2 x 11 +
    /// 2 bits that make double rounding harmless); `None` if `name` is none.
    fn valu16(&mut self, inst: &Inst, name: &str, guard: Option<V>) -> Option<Result<(), Why>> {
        const FLOAT2: &[&str] = &[
            "v_add_f16",
            "v_sub_f16",
            "v_subrev_f16",
            "v_mul_f16",
            "v_max_f16",
            "v_min_f16",
        ];
        const FLOAT3: &[&str] = &[
            "v_fma_f16",
            "v_mad_f16",
            "v_fmac_f16",
            "v_mac_f16",
            "v_max3_f16",
            "v_min3_f16",
            "v_med3_f16",
        ];
        const FLOAT1: &[&str] = &[
            "v_rcp_f16",
            "v_sqrt_f16",
            "v_rsq_f16",
            "v_log_f16",
            "v_exp_f16",
            "v_fract_f16",
            "v_trunc_f16",
            "v_floor_f16",
            "v_ceil_f16",
            "v_rndne_f16",
            "v_sin_f16",
            "v_cos_f16",
            "v_cvt_f16_u16",
            "v_cvt_f16_i16",
        ];
        const INT: &[&str] = &[
            "v_add_nc_u16",
            "v_mad_i16",
            "v_mad_u16",
            "v_min3_i16",
            "v_min3_u16",
            "v_max3_i16",
            "v_max3_u16",
            "v_med3_i16",
            "v_med3_u16",
            "v_sub_nc_u16",
            "v_add_nc_i16",
            "v_sub_nc_i16",
            "v_subrev_nc_u16",
            "v_mul_lo_u16",
            "v_lshlrev_b16",
            "v_lshrrev_b16",
            "v_ashrrev_i16",
            "v_max_u16",
            "v_max_i16",
            "v_min_u16",
            "v_min_i16",
            "v_cvt_u16_f16",
            "v_cvt_i16_f16",
            "v_cvt_norm_i16_f16",
            "v_cvt_norm_u16_f16",
        ];
        let all = [FLOAT1, FLOAT2, FLOAT3, INT];
        if !all.iter().any(|l| l.contains(&name))
            && !matches!(name, "v_cvt_f16_f32" | "v_cvt_f32_f16")
        {
            return None;
        }
        Some(self.valu16_op(inst, name, guard, FLOAT1, FLOAT2, FLOAT3))
    }

    #[allow(clippy::too_many_arguments)]
    fn valu16_op(
        &mut self,
        inst: &Inst,
        name: &str,
        guard: Option<V>,
        float1: &[&str],
        float2: &[&str],
        float3: &[&str],
    ) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let i32t = self.s.t.i32;
        let f32t = self.s.t.f32;
        if name == "v_cvt_f32_f16" {
            let f = self.half_fsrc(inst, 0)?;
            return self.write_f(inst, f, guard);
        }
        let float_result = |l: &mut Self, mut f: V| -> Result<(), Why> {
            if v.omod != 0 {
                return Err("an output modifier on a 16-bit operation");
            }
            if v.clamp {
                f = l.unary(7, f);
            }
            let h =
                l.s.op(l.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
            l.write_half(inst, h, guard)
        };
        if name == "v_cvt_f16_f32" {
            let f = self.fsrc(inst, 0)?;
            return float_result(self, f);
        }
        if float2.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let r = match name {
                "v_add_f16" => self.s.m.bin(c, Bin::Add, a, b),
                "v_sub_f16" => self.s.m.bin(c, Bin::Sub, a, b),
                "v_subrev_f16" => self.s.m.bin(c, Bin::Sub, b, a),
                "v_mul_f16" => self.s.m.bin(c, Bin::Mul, a, b),
                "v_max_f16" => self.max_min(true, a, b),
                _ => self.max_min(false, a, b),
            };
            return float_result(self, r);
        }
        if float3.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let acc = if matches!(name, "v_fmac_f16" | "v_mac_f16") {
                // The accumulator is the destination's half that is written.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("an accumulator")?)?;
                let x = if v.opsel >> 3 & 1 != 0 {
                    let k16 = self.s.m.i32(16);
                    self.s.m.bin(c, Bin::LShr, x, k16)
                } else {
                    x
                };
                let m = self.s.m.i32(0xFFFF);
                let x = self.s.m.bin(c, Bin::And, x, m);
                self.s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
            } else {
                self.half_fsrc(inst, 2)?
            };
            let r = match name {
                // shortcut: fused f16 multiply-adds as FMad in f32 (the
                // product is exact; the sum rounds twice).
                "v_fma_f16" | "v_fmac_f16" | "v_mad_f16" | "v_mac_f16" => self.fmad(a, b, acc),
                "v_max3_f16" => {
                    let t = self.max_min(true, a, b);
                    self.max_min(true, t, acc)
                }
                "v_min3_f16" => {
                    let t = self.max_min(false, a, b);
                    self.max_min(false, t, acc)
                }
                _ => self.med3(a, b, acc),
            };
            return float_result(self, r);
        }
        if float1.contains(&name) {
            let r = match name {
                "v_cvt_f16_u16" | "v_cvt_f16_i16" => {
                    let mut x = self.half_src(inst, 0)?;
                    let op = if name == "v_cvt_f16_i16" {
                        let k16 = self.s.m.i32(16);
                        let t = self.s.m.bin(c, Bin::Shl, x, k16);
                        x = self.s.m.bin(c, Bin::AShr, t, k16);
                        Cast::SiToFp
                    } else {
                        Cast::UiToFp
                    };
                    self.s.m.cast(c, op, x, f32t)
                }
                "v_rcp_f16" => {
                    let a = self.half_fsrc(inst, 0)?;
                    let one = self.s.m.f32(1.0f32.to_bits());
                    self.s.m.bin(c, Bin::SDiv, one, a)
                }
                _ => {
                    let a = self.half_fsrc(inst, 0)?;
                    let op = match name {
                        "v_sqrt_f16" => 24,
                        "v_rsq_f16" => 25,
                        "v_log_f16" => 23,
                        "v_exp_f16" => 21,
                        "v_fract_f16" => 22,
                        "v_trunc_f16" => 29,
                        "v_floor_f16" => 27,
                        "v_ceil_f16" => 28,
                        "v_rndne_f16" => 26,
                        "v_sin_f16" => 13,
                        _ => 12,
                    };
                    match op {
                        12 | 13 => self.trig_turns(a, op == 12),
                        // Any f16 in (-2^-11, 0) has fract >= 1 - 2^-12; clamp to 0x3bff (1 - 2^-11).
                        22 => self.fract(a, 1.0 - 1.0 / 4096.0, 1.0 - 1.0 / 2048.0),
                        _ => self.unary(op, a),
                    }
                }
            };
            return float_result(self, r);
        }
        if matches!(name, "v_cvt_norm_i16_f16" | "v_cvt_norm_u16_f16") {
            if v.clamp || v.omod != 0 {
                return Err("modifiers on a 16-bit integer operation");
            }
            let r = self.cvt_norm16(inst, name == "v_cvt_norm_i16_f16")?;
            return self.write_half(inst, r, guard);
        }
        // 16-bit integers: computed on the low halves, the low 16 bits kept.
        let saturates = matches!(name, "v_mad_i16" | "v_mad_u16");
        if v.clamp && !saturates || v.omod != 0 || v.abs != 0 || v.neg != 0 {
            return Err("modifiers on a 16-bit integer operation");
        }
        let r = match name {
            "v_cvt_u16_f16" | "v_cvt_i16_f16" => {
                // Toward zero, saturating at the 16-bit range (infinities
                // included); NaN gives 0. Clamped in f32 first, where every
                // f16 is exact, so the conversion never leaves i32's range.
                let a = self.half_fsrc(inst, 0)?;
                let (lo, hi) = if name == "v_cvt_u16_f16" {
                    (0.0f32, 65535.0f32)
                } else {
                    (-32768.0, 32767.0)
                };
                let (lo, hi) = (self.s.m.f32(lo.to_bits()), self.s.m.f32(hi.to_bits()));
                let t = self.binary(35, a, lo);
                let t = self.binary(36, t, hi);
                let r = self.s.m.cast(c, Cast::FpToSi, t, i32t);
                let nan = self.s.m.cmp(c, pred::UNO, a, a);
                let z = self.s.m.i32(0);
                self.s.m.select(c, nan, z, r)
            }
            _ => {
                let signed = name.ends_with("_i16");
                let read = |l: &mut Self, i| -> Result<V, Why> {
                    let x = l.half_src(inst, i)?;
                    Ok(if signed { l.sign_extend(x, 16) } else { x })
                };
                let (a, b) = (read(self, 0)?, read(self, 1)?);
                let k15 = self.s.m.i32(15);
                match name {
                    "v_add_nc_u16" | "v_add_nc_i16" => self.s.m.bin(c, Bin::Add, a, b),
                    "v_sub_nc_u16" | "v_sub_nc_i16" => self.s.m.bin(c, Bin::Sub, a, b),
                    "v_subrev_nc_u16" => self.s.m.bin(c, Bin::Sub, b, a),
                    "v_mul_lo_u16" => self.s.m.bin(c, Bin::Mul, a, b),
                    "v_mad_i16" | "v_mad_u16" => {
                        let d = read(self, 2)?;
                        let m = self.s.m.bin(c, Bin::Mul, a, b);
                        self.s.m.bin(c, Bin::Add, m, d)
                    }
                    "v_min3_i16" | "v_min3_u16" | "v_max3_i16" | "v_max3_u16"
                    | "v_med3_i16" | "v_med3_u16" => {
                        let d = read(self, 2)?;
                        self.int3(&name[2..6], signed, a, b, d)
                    }
                    "v_lshlrev_b16" | "v_lshrrev_b16" | "v_ashrrev_i16" => {
                        // src0 is the shift (4 bits), src1 the value.
                        let sh = self.s.m.bin(c, Bin::And, a, k15);
                        let op = match name {
                            "v_lshlrev_b16" => Bin::Shl,
                            "v_lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        self.s.m.bin(c, op, b, sh)
                    }
                    _ => {
                        let p = match name {
                            "v_max_u16" => pred::IUGT,
                            "v_min_u16" => pred::IULT,
                            "v_max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, a, b);
                        self.s.m.select(c, t, a, b)
                    }
                }
            }
        };
        let r = if v.clamp {
            let signed = name.ends_with("_i16");
            let (lo, hi) = if signed { (-32768, 32767) } else { (0, 65535) };
            self.clamp_int(r, 32, lo, hi, signed)
        } else {
            r
        };
        self.write_half(inst, r, guard)
    }

    /// `v_cvt_norm_i16_f16` / `v_cvt_norm_u16_f16`: the f16 source (with its
    /// abs/neg modifiers) clamped to [-1, 1] or [0, 1], times 32767 or 65535,
    /// rounded to nearest even once from the exact product; NaN gives 0.
    /// Worked on the f16's bits in integers, because in f32 the product
    /// (up to 26 significant bits) would round before the rounding to an
    /// integer.
    fn cvt_norm16(&mut self, inst: &Inst, signed: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let mut h = self.half_src(inst, 0)?;
        if v.abs & 1 != 0 {
            let m = k(self, 0x7FFF);
            h = self.s.m.bin(c, Bin::And, h, m);
        }
        if v.neg & 1 != 0 {
            let m = k(self, 0x8000);
            h = self.s.m.bin(c, Bin::Xor, h, m);
        }
        let scale = if signed { 0x7FFF } else { 0xFFFF };
        let (k0, k1, k10) = (k(self, 0), k(self, 1), k(self, 10));
        let mag_mask = k(self, 0x7FFF);
        let mag = self.s.m.bin(c, Bin::And, h, mag_mask);
        let k15 = k(self, 15);
        let neg = self.s.m.bin(c, Bin::LShr, h, k15);
        let neg = self.s.m.cmp(c, pred::NE, neg, k0);
        // |x| < 1: x = m * 2^-s with m < 2^11 and 11 <= s <= 24.
        let exp = self.s.m.bin(c, Bin::LShr, mag, k10);
        let man_mask = k(self, 0x3FF);
        let man = self.s.m.bin(c, Bin::And, mag, man_mask);
        let implicit = k(self, 0x400);
        let normal = self.s.m.bin(c, Bin::Or, man, implicit);
        let sub = self.s.m.cmp(c, pred::EQ, exp, k0);
        let m = self.s.m.select(c, sub, man, normal);
        let k25 = k(self, 25);
        let s_norm = self.s.m.bin(c, Bin::Sub, k25, exp);
        let k24 = k(self, 24);
        let sh = self.s.m.select(c, sub, k24, s_norm);
        let ks = k(self, scale);
        let p = self.s.m.bin(c, Bin::Mul, m, ks);
        let q = self.s.m.bin(c, Bin::LShr, p, sh);
        let one_s = self.s.m.bin(c, Bin::Shl, k1, sh);
        let low_mask = self.s.m.bin(c, Bin::Sub, one_s, k1);
        let rem = self.s.m.bin(c, Bin::And, p, low_mask);
        let half = self.s.m.bin(c, Bin::LShr, one_s, k1);
        let above = self.s.m.cmp(c, pred::IUGT, rem, half);
        let tie = self.s.m.cmp(c, pred::EQ, rem, half);
        let odd = self.s.m.bin(c, Bin::And, q, k1);
        let odd = self.s.m.cmp(c, pred::NE, odd, k0);
        let tie_up = self.s.m.bin(c, Bin::And, tie, odd);
        let up = self.s.m.bin(c, Bin::Or, above, tie_up);
        let r = self.s.m.select(c, up, k1, k0);
        let r = self.s.m.bin(c, Bin::Add, q, r);
        // |x| >= 1 (and infinity) saturates; NaN gives 0.
        let one_h = k(self, 0x3C00);
        let big = self.s.m.cmp(c, pred::IUGE, mag, one_h);
        let r = self.s.m.select(c, big, ks, r);
        let inf = k(self, 0x7C00);
        let nan = self.s.m.cmp(c, pred::IUGT, mag, inf);
        let r = self.s.m.select(c, nan, k0, r);
        let r = if signed {
            let n = self.s.m.bin(c, Bin::Sub, k0, r);
            self.s.m.select(c, neg, n, r)
        } else {
            self.s.m.select(c, neg, k0, r)
        };
        let m16 = k(self, 0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, r, m16))
    }

    /// Half `high` of source `k` of a packed (VOP3P) operation, in the low
    /// 16 bits: op_sel picks the half the low result reads, op_sel_hi the
    /// half the high result reads. Packed integer instructions see the f32
    /// bits of an inline float constant.
    fn packed_src(&mut self, inst: &Inst, k: usize, high: bool, float: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let sel = if high { v.opsel_hi } else { v.opsel } >> k & 1 != 0;
        let x = if float {
            self.f16_operand(inst, &op)?
        } else {
            self.operand(inst, &op)?
        };
        let c = self.cur;
        let x = if sel {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// Packed 16-bit operations (`v_pk_*`): each half computed as the
    /// 16-bit operation does, f16 through f32 as in [`Self::valu16`].
    /// `v_pk_fmac_f16`: each half of the destination plus the product of the
    /// sources' halves.
    // shortcut: FMad in f32, as v_fma_f16 (a second rounding).
    fn packed_fmac(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a packed half FMA's destination");
        };
        if v.src.iter().any(|o| matches!(o, Operand::Const(_) | Operand::Literal(_)))
            || v.sdwa.is_some()
            || v.dpp.is_some()
        {
            return Err("a packed half FMA of constants");
        }
        let c = self.cur;
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let old = inst.read(Reg::V(reg)).ok_or("an unread destination")?;
        let words = [self.operand(inst, &v.src[0])?, self.operand(inst, &v.src[1])?, self.i32_of(old)?];
        let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
        let mut halves = Vec::new();
        for high in [false, true] {
            let xs: Vec<V> = words
                .iter()
                .map(|&w| {
                    let w = if high { self.s.m.bin(c, Bin::LShr, w, k16) } else { w };
                    let w = self.s.m.bin(c, Bin::And, w, low);
                    self.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[w])
                })
                .collect();
            let f = self.fmad(xs[0], xs[1], xs[2]);
            halves.push(self.s.op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]));
        }
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let lo = self.s.m.bin(c, Bin::And, halves[0], low);
        let x = self.s.m.bin(c, Bin::Or, lo, hi);
        self.write_i(inst, x, guard)
    }

    fn packed16(&mut self, inst: &Inst, op: &str, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let float = op.ends_with("_f16");
        let n = match op {
            "fma_f16" | "mad_u16" | "mad_i16" => 3,
            "add_f16" | "mul_f16" | "min_f16" | "max_f16" | "add_u16" | "sub_u16" | "add_i16"
            | "sub_i16" | "mul_lo_u16" | "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16"
            | "max_u16" | "min_u16" | "max_i16" | "min_i16" => 2,
            _ => return Err("a packed 16-bit operation"),
        };
        if v.omod != 0 || !float && (v.clamp || v.neg != 0 || v.neg_hi != 0) {
            return Err("modifiers on a packed integer operation");
        }
        let c = self.cur;
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let mut halves = Vec::new();
        for high in [false, true] {
            let mut xs = Vec::new();
            for k in 0..n {
                let h = self.packed_src(inst, k, high, float)?;
                if float {
                    let mut f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    let neg = if high { v.neg_hi } else { v.neg };
                    if neg >> k & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        f = self.s.m.bin(c, Bin::Sub, z, f);
                    }
                    xs.push(f);
                } else {
                    xs.push(h);
                }
            }
            let r = if float {
                let mut f = match op {
                    "add_f16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "mul_f16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "min_f16" => self.max_min(false, xs[0], xs[1]),
                    "max_f16" => self.max_min(true, xs[0], xs[1]),
                    // shortcut: FMad in f32, as v_fma_f16.
                    _ => self.fmad(xs[0], xs[1], xs[2]),
                };
                if v.clamp {
                    f = self.unary(7, f);
                }
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
            } else {
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if op.ends_with("_i16") {
                    for x in &mut xs {
                        *x = sext(self, *x);
                    }
                }
                let k15 = self.s.m.i32(15);
                match op {
                    "add_u16" | "add_i16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "sub_u16" | "sub_i16" => self.s.m.bin(c, Bin::Sub, xs[0], xs[1]),
                    "mul_lo_u16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "mad_u16" | "mad_i16" => {
                        let m = self.s.m.bin(c, Bin::Mul, xs[0], xs[1]);
                        self.s.m.bin(c, Bin::Add, m, xs[2])
                    }
                    "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16" => {
                        let sh = self.s.m.bin(c, Bin::And, xs[0], k15);
                        let o = match op {
                            "lshlrev_b16" => Bin::Shl,
                            "lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        let x = if op == "ashrrev_i16" {
                            sext(self, xs[1])
                        } else {
                            xs[1]
                        };
                        self.s.m.bin(c, o, x, sh)
                    }
                    _ => {
                        let p = match op {
                            "max_u16" => pred::IUGT,
                            "min_u16" => pred::IULT,
                            "max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, xs[0], xs[1]);
                        self.s.m.select(c, t, xs[0], xs[1])
                    }
                }
            };
            let m = self.s.m.i32(0xFFFF);
            halves.push(self.s.m.bin(c, Bin::And, r, m));
        }
        let k16 = self.s.m.i32(16);
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a packed result outside a VGPR");
        };
        self.write(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    /// The sign or carry bit above the 64-bit multiply-add is returned in
    /// SDST. LLVM's AMDGPURegisterBankInfo describes the signed form as the
    /// low bit of sign(product) + sign(accumulator) + unsigned carry-out.
    fn wide_mad(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else { unreachable!() };
        if v.clamp {
            return Err("clamped 64-bit multiply-add carry semantics");
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("a wide multiply-add result") };
        let signed = inst.name() == "v_mad_i64_i32";
        let (a, b) = (self.src(inst, 0)?, self.src(inst, 1)?);
        let (a, b) = (self.extend64(a, signed), self.extend64(b, signed));
        let acc = self.src64(inst, 2)?;
        let c = self.cur;
        let prod = self.s.m.bin(c, Bin::Mul, a, b);
        let sum = self.s.m.bin(c, Bin::Add, prod, acc);
        let mut carry = self.s.m.cmp(c, pred::IULT, sum, prod);
        if signed {
            let t = self.s.m.int(64);
            let z = self.s.m.int_const(t, 0);
            let pneg = self.s.m.cmp(c, pred::SLT, prod, z);
            let aneg = self.s.m.cmp(c, pred::SLT, acc, z);
            let signs = self.s.m.bin(c, Bin::Xor, pneg, aneg);
            carry = self.s.m.bin(c, Bin::Xor, carry, signs);
        }
        if let Some(Operand::Scalar { code, .. }) = v.carry
            && let Some(w) = inst.write(Reg::S(code))
        {
            self.lane_bits.insert(w, carry);
        }
        let (lo, hi) = self.split64(sum);
        self.write(inst, Reg::V(reg), lo, guard);
        self.write(inst, Reg::V(reg + 1), hi, guard);
        for &(_, w) in &inst.writes {
            self.vals
                .entry(w)
                .or_insert(Val::Missing("a lane mask read as data"));
        }
        Ok(())
    }

    fn int_field(&mut self, x: V, shift: u32, width: u32) -> V {
        if width == 32 {
            return x;
        }
        let sh = self.s.m.i32(shift);
        let mask = self.s.m.i32((1 << width) - 1);
        let x = self.s.m.bin(self.cur, Bin::LShr, x, sh);
        self.s.m.bin(self.cur, Bin::And, x, mask)
    }

    fn int3(&mut self, op: &str, signed: bool, a: V, b: V, d: V) -> V {
        let (lt, gt) = if signed { (pred::SLT, pred::SGT) } else { (pred::IULT, pred::IUGT) };
        match op {
            "min3" => {
                let t = self.int_pick(lt, a, b);
                self.int_pick(lt, t, d)
            }
            "max3" => {
                let t = self.int_pick(gt, a, b);
                self.int_pick(gt, t, d)
            }
            _ => {
                let lo = self.int_pick(lt, a, b);
                let hi = self.int_pick(gt, a, b);
                let t = self.int_pick(lt, hi, d);
                self.int_pick(gt, lo, t)
            }
        }
    }

    /// Integer dot products, including the optional output saturation in
    /// VOP3P (LLVM AMDGPUUsage's sdot/udot intrinsics).
    fn integer_dot(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else { unreachable!() };
        let name = inst.name();
        let implicit = name == "v_dot4c_i32_i8";
        let width = if name.starts_with("v_dot2") { 16 } else if name.starts_with("v_dot4") { 8 } else { 4 };
        let signed = name.contains("_i32");
        if v.neg != 0 || v.neg_hi != 0 || v.abs != 0 || v.omod != 0 || v.sdwa.is_some() {
            return Err("modifiers on an integer dot product");
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("an integer dot product result") };
        let c = self.cur;
        let a = self.src(inst, 0)?;
        let b = self.src(inst, 1)?;
        let d = if implicit {
            self.i32_of(inst.read(Reg::V(reg)).ok_or("a dot product accumulator")?)?
        } else {
            self.src(inst, 2)?
        };
        let mut sum = if v.clamp { self.extend64(d, signed) } else { d };
        for i in 0..32 / width {
            // dot2 has independent low/high half selects for each input.
            let (sa, sb) = if width == 16 {
                let sel = if i == 0 { v.opsel } else { v.opsel_hi };
                (u32::from(sel & 1) * 16, u32::from(sel >> 1 & 1) * 16)
            } else {
                (i * width, i * width)
            };
            let (mut a, mut b) = (self.int_field(a, sa, width), self.int_field(b, sb, width));
            if signed {
                a = self.sign_extend(a, width);
                b = self.sign_extend(b, width);
            }
            if v.clamp {
                a = self.extend64(a, signed);
                b = self.extend64(b, signed);
            }
            let p = self.s.m.bin(c, Bin::Mul, a, b);
            sum = self.s.m.bin(c, Bin::Add, sum, p);
        }
        if v.clamp {
            let (min, max) = if signed { (i32::MIN as i64, i32::MAX as i64) } else { (0, u32::MAX as i64) };
            sum = self.clamp_int(sum, 64, min, max, signed);
            sum = self.s.m.cast(c, Cast::Trunc, sum, self.s.t.i32);
        }
        self.write(inst, Reg::V(reg), sum, guard);
        Ok(())
    }

    /// Sum of the unsigned absolute differences of the packed elements.
    fn sad(&mut self, a: V, b: V, width: u32, masked: bool) -> V {
        let c = self.cur;
        let zero = self.s.m.i32(0);
        let mut sum = zero;
        for offset in (0..32).step_by(width as usize) {
            let x = self.int_field(a, offset, width);
            let y = self.int_field(b, offset, width);
            let ge = self.s.m.cmp(c, pred::IUGE, x, y);
            let xy = self.s.m.bin(c, Bin::Sub, x, y);
            let yx = self.s.m.bin(c, Bin::Sub, y, x);
            let mut diff = self.s.m.select(c, ge, xy, yx);
            if masked {
                let skip = self.s.m.cmp(c, pred::EQ, y, zero);
                diff = self.s.m.select(c, skip, zero, diff);
            }
            sum = self.s.m.bin(c, Bin::Add, sum, diff);
        }
        sum
    }

    fn integer(&mut self, inst: &Inst, name: &'static str) -> Result<V, Why> {
        let c = self.cur;
        let s = |l: &mut Self, k| l.src(inst, k);
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let m31 = |l: &mut Self, x: V| {
            let k31 = l.s.m.i32(31);
            l.s.m.bin(c, Bin::And, x, k31)
        };
        Ok(match name {
            "v_and_b32" | "v_or_b32" | "v_xor_b32" | "v_add_nc_u32" | "v_add_nc_i32"
            | "v_sub_nc_u32" | "v_sub_nc_i32" | "v_subrev_nc_u32" | "v_mul_lo_u32"
            | "v_mul_lo_i32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (op, a, b) = match name {
                    "v_and_b32" => (Bin::And, a, b),
                    "v_or_b32" => (Bin::Or, a, b),
                    "v_xor_b32" => (Bin::Xor, a, b),
                    "v_add_nc_u32" | "v_add_nc_i32" => (Bin::Add, a, b),
                    "v_sub_nc_u32" | "v_sub_nc_i32" => (Bin::Sub, a, b),
                    "v_subrev_nc_u32" => (Bin::Sub, b, a),
                    _ => (Bin::Mul, a, b),
                };
                self.s.m.bin(c, op, a, b)
            }
            "v_xnor_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let x = self.s.m.bin(c, Bin::Xor, a, b);
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, x, ones)
            }
            "v_not_b32" => {
                let a = s(self, 0)?;
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, a, ones)
            }
            "v_lshlrev_b32" | "v_lshrrev_b32" | "v_ashrrev_i32" => {
                let (sh, a) = (s(self, 0)?, s(self, 1)?);
                let sh = m31(self, sh);
                let op = match name {
                    "v_lshlrev_b32" => Bin::Shl,
                    "v_lshrrev_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                self.s.m.bin(c, op, a, sh)
            }
            "v_lshl_add_u32" | "v_lshl_or_b32" | "v_add_lshl_u32" | "v_and_or_b32"
            | "v_or3_b32" | "v_xad_u32" | "v_add3_u32" | "v_xor3_b32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                match name {
                    "v_lshl_add_u32" | "v_lshl_or_b32" => {
                        let b = m31(self, b);
                        let t = self.s.m.bin(c, Bin::Shl, a, b);
                        let op = if name == "v_lshl_add_u32" {
                            Bin::Add
                        } else {
                            Bin::Or
                        };
                        self.s.m.bin(c, op, t, d)
                    }
                    "v_add_lshl_u32" => {
                        let t = self.s.m.bin(c, Bin::Add, a, b);
                        let d = m31(self, d);
                        self.s.m.bin(c, Bin::Shl, t, d)
                    }
                    _ => {
                        let (o1, o2) = match name {
                            "v_and_or_b32" => (Bin::And, Bin::Or),
                            "v_or3_b32" => (Bin::Or, Bin::Or),
                            "v_xad_u32" => (Bin::Xor, Bin::Add),
                            "v_add3_u32" => (Bin::Add, Bin::Add),
                            _ => (Bin::Xor, Bin::Xor),
                        };
                        let t = self.s.m.bin(c, o1, a, b);
                        self.s.m.bin(c, o2, t, d)
                    }
                }
            }
            "v_mul_u32_u24" | "v_mul_i32_i24" | "v_mul_hi_i32_i24" | "v_mul_hi_u32_u24" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (sh, op) = (
                    k(self, 8),
                    if name.ends_with("i24") {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                if name.starts_with("v_mul_hi") {
                    self.mul_hi(a, b, name.ends_with("i24"))
                } else {
                    self.s.m.bin(c, Bin::Mul, a, b)
                }
            }
            "v_sad_u8" | "v_sad_hi_u8" | "v_sad_u16" | "v_sad_u32" | "v_msad_u8" => {
                // RDNA 2 ISA, VOP3A: accumulation wraps in the dword.
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let width = if name.ends_with("u8") { 8 } else if name.ends_with("u16") { 16 } else { 32 };
                let sum = self.sad(a, b, width, name == "v_msad_u8");
                let sum = if name == "v_sad_hi_u8" {
                    let k16 = k(self, 16);
                    self.s.m.bin(c, Bin::Shl, sum, k16)
                } else {
                    sum
                };
                self.s.m.bin(c, Bin::Add, sum, d)
            }
            "v_lerp_u8" => {
                // RDNA 2 ISA, VOP3A: low bit of each byte of S2 rounds a tie up.
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let one = k(self, 1);
                let mut result = k(self, 0);
                for shift in [0, 8, 16, 24] {
                    let x = self.int_field(a, shift, 8);
                    let y = self.int_field(b, shift, 8);
                    let round = self.int_field(d, shift, 1);
                    let sum = self.s.m.bin(c, Bin::Add, x, y);
                    let sum = self.s.m.bin(c, Bin::Add, sum, round);
                    let avg = self.s.m.bin(c, Bin::LShr, sum, one);
                    let sh = k(self, shift);
                    let field = self.s.m.bin(c, Bin::Shl, avg, sh);
                    result = self.s.m.bin(c, Bin::Or, result, field);
                }
                result
            }
            "v_bfe_u32" | "v_bfe_i32" => {
                let (a, off, w) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let w = m31(self, w);
                self.bfe(a, off, w, name == "v_bfe_i32")
            }
            "v_bfi_b32" => {
                let (m, a, b) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let ones = k(self, u32::MAX);
                let nm = self.s.m.bin(c, Bin::Xor, m, ones);
                let x = self.s.m.bin(c, Bin::And, m, a);
                let y = self.s.m.bin(c, Bin::And, nm, b);
                self.s.m.bin(c, Bin::Or, x, y)
            }
            "v_max_i32" | "v_min_i32" | "v_max_u32" | "v_min_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let p = match name {
                    "v_max_i32" => pred::SGT,
                    "v_min_i32" => pred::SLT,
                    "v_max_u32" => pred::IUGT,
                    _ => pred::IULT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                self.s.m.select(c, t, a, b)
            }
            "v_mul_hi_u32" | "v_mul_hi_i32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                self.mul_hi(a, b, name == "v_mul_hi_i32")
            }
            "v_cvt_pk_u16_u32" | "v_cvt_pk_i16_i32" => {
                // Each source saturated to 16 bits: src0 into the low half,
                // src1 into the high one.
                let unsigned = name == "v_cvt_pk_u16_u32";
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let mut halves = [a, b];
                for x in &mut halves {
                    *x = if unsigned {
                        let max = k(self, 0xFFFF);
                        let over = self.s.m.cmp(c, pred::IUGT, *x, max);
                        self.s.m.select(c, over, max, *x)
                    } else {
                        let (lo, hi, m) = (k(self, 0xFFFF_8000), k(self, 0x7FFF), k(self, 0xFFFF));
                        let below = self.s.m.cmp(c, pred::SLT, *x, lo);
                        let t = self.s.m.select(c, below, lo, *x);
                        let above = self.s.m.cmp(c, pred::SGT, t, hi);
                        let t = self.s.m.select(c, above, hi, t);
                        self.s.m.bin(c, Bin::And, t, m)
                    };
                }
                let k16 = k(self, 16);
                let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
                self.s.m.bin(c, Bin::Or, halves[0], hi)
            }
            "v_dot2_i32_i16" | "v_dot2_u32_u16" | "v_dot4_i32_i8" | "v_dot4_u32_u8"
            | "v_dot8_u32_u4" => {
                // The products of the packed fields of src0 and src1, summed
                // into src2, wrapping (clamp is refused with the other
                // integer output modifiers).
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                if v.opsel != 0 || v.neg_hi != 0 {
                    return Err("dot product operand selects");
                }
                let (bits, signed) = match name {
                    "v_dot2_i32_i16" => (16, true),
                    "v_dot2_u32_u16" => (16, false),
                    "v_dot4_i32_i8" => (8, true),
                    "v_dot4_u32_u8" => (8, false),
                    _ => (4, false),
                };
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let mut sum = s(self, 2)?;
                let op = if signed { Bin::AShr } else { Bin::LShr };
                for i in 0..32 / bits {
                    // Field i to the top, then down again (extending it).
                    let (up, down) = (k(self, 32 - bits * (i + 1)), k(self, 32 - bits));
                    let x = self.s.m.bin(c, Bin::Shl, a, up);
                    let x = self.s.m.bin(c, op, x, down);
                    let y = self.s.m.bin(c, Bin::Shl, b, up);
                    let y = self.s.m.bin(c, op, y, down);
                    let p = self.s.m.bin(c, Bin::Mul, x, y);
                    sum = self.s.m.bin(c, Bin::Add, sum, p);
                }
                sum
            }
            "v_alignbit_b32" | "v_alignbyte_b32" => {
                // The low dword of {a, b} >> shift.
                let (a, b, sh) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let sh = if name == "v_alignbyte_b32" {
                    let three = k(self, 3);
                    let t = self.s.m.bin(c, Bin::And, sh, three);
                    let k3 = k(self, 3);
                    self.s.m.bin(c, Bin::Shl, t, k3)
                } else {
                    m31(self, sh)
                };
                let k32 = k(self, 32);
                let lo = self.s.m.bin(c, Bin::LShr, b, sh);
                let left = self.s.m.bin(c, Bin::Sub, k32, sh);
                let left = m31(self, left);
                let hi = self.s.m.bin(c, Bin::Shl, a, left);
                let zero = k(self, 0);
                let none = self.s.m.cmp(c, pred::EQ, sh, zero);
                let hi = self.s.m.select(c, none, zero, hi);
                self.s.m.bin(c, Bin::Or, lo, hi)
            }
            "v_bfrev_b32" => {
                let a = s(self, 0)?;
                self.s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a])
            }
            "v_bcnt_u32_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[a]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            "v_mad_u32_u24" | "v_mad_i32_i24" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let (sh, op) = (
                    k(self, 8),
                    if name == "v_mad_i32_i24" {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                let m = self.s.m.bin(c, Bin::Mul, a, b);
                self.s.m.bin(c, Bin::Add, m, d)
            }
            "v_min3_i32" | "v_max3_i32" | "v_min3_u32" | "v_max3_u32" | "v_med3_i32"
            | "v_med3_u32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                self.int3(&name[2..6], name.ends_with("i32"), a, b, d)
            }
            "v_perm_b32" => {
                // Byte i of the result is byte S of {src0, src1} (src1 the
                // low dword) for selector byte S below 8; 8-11 repeat the
                // sign of bytes 1 and 3 of src1, then of src0; 12 is 0x00,
                // higher 0xFF.
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                let sel = match v.src.get(2) {
                    Some(&Operand::Literal(x)) => x,
                    Some(&Operand::Const(k)) => crate::resource::inline(k).ok_or(name)?,
                    _ => {
                        // A computed selector: each byte picked at run time.
                        let (a, b, sv) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                        let mut r = k(self, 0);
                        for i in 0..4u32 {
                            let (sh, ff) = (k(self, 8 * i), k(self, 0xFF));
                            let sel = self.s.m.bin(c, Bin::LShr, sv, sh);
                            let sel = self.s.m.bin(c, Bin::And, sel, ff);
                            // Select a dword and a byte within it.
                            // Later selects handle codes 8 and above.
                            let four = k(self, 4);
                            let zero = k(self, 0);
                            let high = self.s.m.bin(c, Bin::And, sel, four);
                            let high = self.s.m.cmp(c, pred::NE, high, zero);
                            let word = self.s.m.select(c, high, a, b);
                            let three = k(self, 3);
                            let index = self.s.m.bin(c, Bin::And, sel, three);
                            let bits = self.s.m.bin(c, Bin::Shl, index, three);
                            let x = self.s.m.bin(c, Bin::LShr, word, bits);
                            let mut byte = self.s.m.bin(c, Bin::And, x, ff);
                            // 8-11: the sign of bytes 1, 3 of src1, then of src0.
                            for (n, src, bit) in [(8, b, 15), (9, b, 31), (10, a, 15), (11, a, 31)]
                            {
                                let left = k(self, 31 - bit);
                                let t = self.s.m.bin(c, Bin::Shl, src, left);
                                let k31 = k(self, 31);
                                let t = self.s.m.bin(c, Bin::AShr, t, k31);
                                let t = self.s.m.bin(c, Bin::And, t, ff);
                                let kn = k(self, n);
                                let is = self.s.m.cmp(c, pred::EQ, sel, kn);
                                byte = self.s.m.select(c, is, t, byte);
                            }
                            let (k12, z) = (k(self, 12), k(self, 0));
                            let is12 = self.s.m.cmp(c, pred::EQ, sel, k12);
                            byte = self.s.m.select(c, is12, z, byte);
                            let above = self.s.m.cmp(c, pred::IUGT, sel, k12);
                            byte = self.s.m.select(c, above, ff, byte);
                            let at = k(self, 8 * i);
                            let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                            r = self.s.m.bin(c, Bin::Or, r, byte);
                        }
                        return Ok(r);
                    }
                };
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let mut r = k(self, 0);
                for i in 0..4 {
                    let pick = sel >> (8 * i) & 0xFF;
                    let byte = match pick {
                        0..=7 => {
                            let (src, n) = if pick < 4 { (b, pick) } else { (a, pick - 4) };
                            let sh = k(self, 8 * n);
                            let t = self.s.m.bin(c, Bin::LShr, src, sh);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        8..=11 => {
                            let src = if pick < 10 { b } else { a };
                            let bit = if pick % 2 == 0 { 15 } else { 31 };
                            let left = k(self, 31 - bit);
                            let t = self.s.m.bin(c, Bin::Shl, src, left);
                            let k31 = k(self, 31);
                            let t = self.s.m.bin(c, Bin::AShr, t, k31);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        12 => continue,
                        _ => k(self, 0xFF),
                    };
                    let at = k(self, 8 * i);
                    let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                    r = self.s.m.bin(c, Bin::Or, r, byte);
                }
                r
            }
            "v_bfm_b32" => {
                // ((1 << src0[4:0]) - 1) << src1[4:0]
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let one = k(self, 1);
                let (a, b) = (m31(self, a), m31(self, b));
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                self.s.m.bin(c, Bin::Shl, m, b)
            }
            "v_cvt_pk_u8_f32" => {
                // src0 rounded to nearest even and saturated to a byte, into
                // byte src1[1:0] of src2.
                let f = self.fsrc(inst, 0)?;
                let (b, d) = (s(self, 1)?, s(self, 2)?);
                let f = self.unary(26, f);
                let (zero, max) = (self.s.m.f32(0), self.s.m.f32(255.0f32.to_bits()));
                let f = self.binary(35, f, zero);
                let f = self.binary(36, f, max);
                let x = self.s.m.cast(c, Cast::FpToUi, f, self.s.t.i32);
                let (three, k3) = (k(self, 3), k(self, 3));
                let sh = self.s.m.bin(c, Bin::And, b, three);
                let sh = self.s.m.bin(c, Bin::Shl, sh, k3);
                let x = self.s.m.bin(c, Bin::Shl, x, sh);
                let ff = k(self, 0xFF);
                let hole = self.s.m.bin(c, Bin::Shl, ff, sh);
                let ones = k(self, u32::MAX);
                let hole = self.s.m.bin(c, Bin::Xor, hole, ones);
                let d = self.s.m.bin(c, Bin::And, d, hole);
                self.s.m.bin(c, Bin::Or, d, x)
            }
            "v_ffbl_b32" => {
                // -1 for zero, as DXIL's first-bit-low.
                let a = s(self, 0)?;
                self.s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 32, &[a])
            }
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                // Bits of the mask's low (high) half below this lane, plus src1:
                // with an all-ones mask, the lane index.
                let b = s(self, 1)?;
                // Mesh lanes are the console's by construction; pixel and
                // vertex lanes are the host's, as their masks are.
                if self.mask_data == MaskData::Host {
                    // Host lanes: the mask must be one (a ballot) or all lanes.
                    let Fields::Valu(v) = &inst.fields else {
                        unreachable!()
                    };
                    if name == "v_mbcnt_lo_u32_b32" && !self.x.native.wave64 {
                        if matches!(v.src.first(), Some(&Operand::Const(193))) {
                            self.s.flags |= flags::WAVE_OPS;
                            let lane = self.lane_index();
                            return Ok(self.s.m.bin(c, Bin::Add, lane, b));
                        }
                        if let Some((lo, hi)) = v
                            .src
                            .first()
                            .map(|op| self.host_mask_pair(inst, op))
                            .transpose()?
                            .flatten()
                        {
                            self.s.flags |= flags::WAVE_OPS;
                            let lane = self.lane_index();
                            let m_lo = self.below_mask(c, lane);
                            let (k32, z) = (k(self, 32), k(self, 0));
                            let t = self.s.m.bin(c, Bin::Sub, lane, k32);
                            let low = self.s.m.cmp(c, pred::IULT, lane, k32);
                            let below_hi = self.s.m.select(c, low, z, t);
                            let m_hi = self.below_mask(c, below_hi);
                            let x_lo = self.s.m.bin(c, Bin::And, lo, m_lo);
                            let x_hi = self.s.m.bin(c, Bin::And, hi, m_hi);
                            let n = self.bcnt1_64(c, x_lo, x_hi);
                            return Ok(self.s.m.bin(c, Bin::Add, n, b));
                        }
                    }
                    let host = match v.src.first() {
                        Some(&Operand::Const(193)) => true,
                        Some(&Operand::Scalar { code, .. }) => {
                            inst.read(Reg::S(code)).is_some_and(|m| {
                                self.x.st.masks.contains_key(&m) || self.x.st.highs.contains_key(&m)
                            })
                        }
                        _ => false,
                    };
                    if !host {
                        return Err("v_mbcnt of a mask that is no ballot (console lane layout)");
                    }
                    self.s.flags |= flags::WAVE_OPS;
                } else if self.s.mesh.is_none() {
                    self.cross_lane(false)?;
                }
                let a = s(self, 0)?;
                let lane = if self.console_cf {
                    self.console_lane().0
                } else {
                    self.lane_index()
                };
                let below = if name == "v_mbcnt_lo_u32_b32" {
                    lane
                } else {
                    let (k32, z) = (k(self, 32), k(self, 0));
                    let t = self.s.m.bin(c, Bin::Sub, lane, k32);
                    let low = self.s.m.cmp(c, pred::IULT, lane, k32);
                    self.s.m.select(c, low, z, t)
                };
                let m = self.below_mask(c, below);
                let x = self.s.m.bin(c, Bin::And, a, m);
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[x]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            _ => return Err(name),
        })
    }

    /// AMD RDNA 2 ISA, VOPC: classify the input bits without converting to a
    /// host float. Conversion can quiet an sNaN or flush a denormal.
    fn fp_class(&mut self, inst: &Inst, format: &str) -> Result<V, Why> {
        let (bits, fraction, exp_mask) = match format {
            "f16" => (16, 10, 0x1F),
            "f32" => (32, 23, 0xFF),
            _ => (64, 52, 0x7FF),
        };
        let c = self.cur;
        let ty = self.s.m.int(bits.max(32));
        let x = self.fp_bits(inst, 0, bits)?;
        let mask = if bits == 16 { self.half_src(inst, 1)? } else { self.src(inst, 1)? };
        let (zero, sh, emax, frac, sign, quiet) = (
            self.s.m.int_const(ty, 0), self.s.m.int_const(ty, fraction),
            self.s.m.int_const(ty, exp_mask), self.s.m.int_const(ty, (1 << fraction) - 1),
            self.s.m.int_const(ty, (1u64 << (bits - 1)) as i64),
            self.s.m.int_const(ty, 1 << (fraction - 1)),
        );
        let e = self.s.m.bin(c, Bin::LShr, x, sh);
        let e = self.s.m.bin(c, Bin::And, e, emax);
        let m = self.s.m.bin(c, Bin::And, x, frac);
        let neg = self.s.m.bin(c, Bin::And, x, sign);
        let q = self.s.m.bin(c, Bin::And, x, quiet);
        let subnormal = self.s.m.cmp(c, pred::EQ, e, zero);
        let nonfinite = self.s.m.cmp(c, pred::EQ, e, emax);
        let no_frac = self.s.m.cmp(c, pred::EQ, m, zero);
        let has_frac = self.s.m.cmp(c, pred::NE, m, zero);
        let neg = self.s.m.cmp(c, pred::NE, neg, zero);
        let quiet = self.s.m.cmp(c, pred::NE, q, zero);
        // Positive classes 6-9, negative classes 11 minus those.
        let zero = self.s.m.i32(0);
        let (one, six, seven, eight, nine, eleven) = (self.s.m.i32(1), self.s.m.i32(6),
            self.s.m.i32(7), self.s.m.i32(8), self.s.m.i32(9), self.s.m.i32(11));
        let denorm_or_zero = self.s.m.select(c, no_frac, six, seven);
        let class = self.s.m.select(c, subnormal, denorm_or_zero, eight);
        let class = self.s.m.select(c, nonfinite, nine, class);
        let mirrored = self.s.m.bin(c, Bin::Sub, eleven, class);
        let class = self.s.m.select(c, neg, mirrored, class);
        let nan = self.s.m.bin(c, Bin::And, nonfinite, has_frac);
        let nan_class = self.s.m.select(c, quiet, one, zero);
        let class = self.s.m.select(c, nan, nan_class, class);
        let bit = self.s.m.bin(c, Bin::LShr, mask, class);
        let bit = self.s.m.bin(c, Bin::And, bit, one);
        Ok(self.s.m.cmp(c, pred::NE, bit, zero))
    }

    /// Compare double encodings in IEEE order using integer operations.
    /// Flipping the sign for nonnegative values and complementing negative
    /// ones makes their unsigned order match their numeric order.
    fn compare_f64(&mut self, inst: &Inst, op: &str) -> Result<V, Why> {
        if matches!(op, "f" | "tru") {
            return Ok(self.s.m.bool(op == "tru"));
        }
        let c = self.cur;
        let (a, b) = (self.fp_bits(inst, 0, 64)?, self.fp_bits(inst, 1, 64)?);
        let ty = self.s.m.int(64);
        let (z, sign, abs, inf, ones) = (self.s.m.int_const(ty, 0), self.s.m.int_const(ty, i64::MIN),
            self.s.m.int_const(ty, i64::MAX), self.s.m.int_const(ty, 0x7FF0_0000_0000_0000),
            self.s.m.int_const(ty, -1));
        let abs_a = self.s.m.bin(c, Bin::And, a, abs);
        let abs_b = self.s.m.bin(c, Bin::And, b, abs);
        let nan_a = self.s.m.cmp(c, pred::IUGT, abs_a, inf);
        let nan_b = self.s.m.cmp(c, pred::IUGT, abs_b, inf);
        let unordered = self.s.m.bin(c, Bin::Or, nan_a, nan_b);
        let yes = self.s.m.bool(true);
        let ordered = self.s.m.bin(c, Bin::Xor, unordered, yes);
        if op == "u" || op == "o" {
            return Ok(if op == "u" { unordered } else { ordered });
        }
        let key = |l: &mut Self, x, magnitude| {
            let zero = l.s.m.cmp(c, pred::EQ, magnitude, z);
            let x = l.s.m.select(c, zero, z, x); // both zeros compare equal
            let neg = l.s.m.cmp(c, pred::SLT, x, z);
            let xor = l.s.m.select(c, neg, ones, sign);
            l.s.m.bin(c, Bin::Xor, x, xor)
        };
        let (a, b) = (key(self, a, abs_a), key(self, b, abs_b));
        let (p, negated) = match op {
            "lt" | "nlt" => (pred::IULT, op == "nlt"),
            "eq" | "neq" => (pred::EQ, op == "neq"),
            "le" | "nle" => (pred::IULE, op == "nle"),
            "gt" | "ngt" => (pred::IUGT, op == "ngt"),
            "lg" | "nlg" => (pred::NE, op == "nlg"),
            "ge" | "nge" => (pred::IUGE, op == "nge"),
            _ => return Err("a double compare predicate"),
        };
        let test = self.s.m.cmp(c, p, a, b);
        let test = self.s.m.bin(c, Bin::And, ordered, test);
        Ok(if negated { self.s.m.bin(c, Bin::Xor, test, yes) } else { test })
    }

    fn compare(&mut self, inst: &Inst, rest: &str) -> Result<(), Why> {
        let (op, ty) = rest.rsplit_once('_').ok_or("a compare")?;
        let c = self.cur;
        let r = match ty {
            "f16" | "f32" | "f64" if op == "class" => self.fp_class(inst, ty)?,
            "f64" => self.compare_f64(inst, op)?,
            "f32" | "f16" => {
                let (a, b) = if ty == "f16" {
                    (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?)
                } else {
                    (self.fsrc(inst, 0)?, self.fsrc(inst, 1)?)
                };
                // `fsrc` may emit two separate bitcasts for the same register,
                // which host CSE later merges into `fcmp ... %f, %f`.
                let same_src = a == b
                    || (self.src0.is_none()
                        && matches!(
                            &inst.fields,
                            Fields::Valu(v)
                                if v.src.len() >= 2
                                    && v.src[0] == v.src[1]
                                    && (v.abs & 1) == ((v.abs >> 1) & 1)
                                    && (v.neg & 1) == ((v.neg >> 1) & 1)
                                    && (v.opsel & 1) == ((v.opsel >> 1) & 1)
                                    && v.sdwa.is_none()
                        ));
                // Some D3D12 drivers (observed on Intel Iris Xe) assume no NaNs
                // for plain `fcmp`, folding `fcmp ord` to true, `fcmp uno` to
                // false (even across distinct registers), and self-comparisons
                // like `fcmp oeq/une x, x` to constants. Use DXIL `IsNaN`
                // (`dx.op.isSpecialFloat.f32`, opcode 8) for ordered/unordered
                // and same-value NaN-sensitive predicates.
                let is_nan = |l: &mut Self, x: V| {
                    l.s.op(c, "isSpecialFloat.f32", Attr::ReadNone, l.s.t.i1, 8, &[x])
                };
                let any_nan = |l: &mut Self, x: V, y: V| {
                    let nx = is_nan(l, x);
                    if same_src {
                        nx
                    } else {
                        let ny = is_nan(l, y);
                        l.s.m.bin(c, Bin::Or, nx, ny)
                    }
                };
                match op {
                    "u" => any_nan(self, a, b),
                    "o" => {
                        let u = any_nan(self, a, b);
                        let yes = self.s.m.bool(true);
                        self.s.m.bin(c, Bin::Xor, u, yes)
                    }
                    "eq" | "le" | "ge" if same_src => {
                        let n = is_nan(self, a);
                        let yes = self.s.m.bool(true);
                        self.s.m.bin(c, Bin::Xor, n, yes)
                    }
                    "neq" | "nge" | "nle" if same_src => is_nan(self, a),
                    // Ordered and not equal, false for a NaN, as `olt | ogt`.
                    // `fcmp one` means the same, but the AMD D3D12 driver
                    // runs it as unordered (true for a NaN) where NVIDIA
                    // and WARP do not; olt and ogt agree on all three.
                    "lg" => {
                        let lt = self.s.m.cmp(c, pred::OLT, a, b);
                        let gt = self.s.m.cmp(c, pred::OGT, a, b);
                        self.s.m.bin(c, Bin::Or, lt, gt)
                    }
                    _ => {
                        let p = match op {
                            "f" => pred::FALSE,
                            "lt" => pred::OLT,
                            "eq" => pred::OEQ,
                            "le" => pred::OLE,
                            "gt" => pred::OGT,
                            "ge" => pred::OGE,
                            "nge" => pred::ULT,
                            "nlg" => pred::UEQ,
                            "ngt" => pred::ULE,
                            "nle" => pred::UGT,
                            "neq" => pred::UNE,
                            "nlt" => pred::UGE,
                            "tru" => pred::TRUE,
                            _ => return Err(inst.name()),
                        };
                        self.s.m.cmp(c, p, a, b)
                    }
                }
            }
            "u64" | "i64" if matches!(op, "f" | "t") => self.s.m.bool(op == "t"),
            "u64" | "i64" => {
                let (a, b) = (self.src64(inst, 0)?, self.src64(inst, 1)?);
                let signed = ty == "i64";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    _ => return Err("a 64-bit compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            "u32" | "i32" | "u16" | "i16" => {
                let (a, b) = if ty.ends_with("16") {
                    let (mut a, mut b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
                    if ty == "i16" {
                        let k16 = self.s.m.i32(16);
                        for x in [&mut a, &mut b] {
                            let t = self.s.m.bin(c, Bin::Shl, *x, k16);
                            *x = self.s.m.bin(c, Bin::AShr, t, k16);
                        }
                    }
                    (a, b)
                } else {
                    (self.src(inst, 0)?, self.src(inst, 1)?)
                };
                let signed = ty == "i32" || ty == "i16";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    "f" | "t" => {
                        self.lane(inst, Some(op == "t"));
                        return Ok(());
                    }
                    _ => return Err("an integer compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            _ => return Err(inst.name()),
        };
        self.lane_bits_for(inst, r);
        Ok(())
    }

    fn lane(&mut self, inst: &Inst, value: Option<bool>) {
        let b = self.s.m.bool(value == Some(true));
        self.lane_bits_for(inst, b);
    }

    fn lane_bits_for(&mut self, inst: &Inst, r: V) {
        let Fields::Valu(v) = &inst.fields else {
            return;
        };
        let code = if inst.name().starts_with("v_cmpx") {
            Some(EXEC)
        } else {
            match v.dst {
                Some(Operand::Scalar { code, .. }) => Some(code),
                _ => None,
            }
        };
        if let Some(lo) = code.and_then(|c| inst.write(Reg::S(c))) {
            self.lane_bits.insert(lo, r);
        }
        for &(_, w) in &inst.writes {
            self.vals
                .entry(w)
                .or_insert(Val::Missing("a lane mask read as data"));
        }
    }
}
