library(ggplot2)
library(dplyr)
library(scales)

# ── data ─────────────────────────────────────────────────────────────────────
csv_path <- "C:/Users/admin/Desktop/My-paper--main/論文/實驗/Case test/TABLE13_FULL_INSGA3.CSV"
out_dir  <- "C:/Users/admin/Desktop/My-paper--main/論文/figures"

df_raw <- read.csv(csv_path, stringsAsFactors = FALSE)

# pivot to wide for fixed-sat lines, keep dynamic (optimal) separately
df_wide <- df_raw %>%
  select(day, n_sat, cost_avg, nv_avg, co2_avg, rt_avg, is_optimal) %>%
  mutate(n_sat = factor(n_sat, levels = c(1, 2, 3),
                        labels = c("Fixed Sat=1", "Fixed Sat=2", "Fixed Sat=3")))

df_dyn <- df_raw %>%
  filter(is_optimal == 1) %>%
  mutate(opt_label = factor(n_sat, levels = c(1, 2, 3),
                            labels = c("Optimal = Sat 1", "Optimal = Sat 2", "Optimal = Sat 3")))

# ── shared theme ──────────────────────────────────────────────────────────────
base_theme <- theme_bw(base_size = 11) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(linetype = "dotted", colour = "grey70"),
    axis.text.x  = element_text(size = 7.5),
    axis.text.y  = element_text(size = 9),
    legend.position   = "right",
    legend.key.width  = unit(1.8, "cm"),
    legend.text       = element_text(size = 8.5),
    legend.title      = element_blank(),
    plot.title        = element_text(size = 11, face = "bold", hjust = 0.5),
    plot.margin       = margin(8, 8, 8, 8)
  )

line_types <- c("Fixed Sat=1" = "dashed",
                "Fixed Sat=2" = "dotted",
                "Fixed Sat=3" = "longdash")
line_cols  <- c("Fixed Sat=1" = "#2288FF",   # 電藍
                "Fixed Sat=2" = "#FF6600",   # 亮橘
                "Fixed Sat=3" = "#22CC44")   # 鮮綠
pt_shapes  <- c("Optimal = Sat 1" = 16,   # filled circle
                "Optimal = Sat 2" = 17,   # filled triangle
                "Optimal = Sat 3" = 15)   # filled square
pt_cols    <- c("Optimal = Sat 1" = "#2288FF",
                "Optimal = Sat 2" = "#FF6600",
                "Optimal = Sat 3" = "#22CC44")

# ── plot factory ──────────────────────────────────────────────────────────────
make_plot <- function(col, y_label, title, out_file, y_fmt = comma) {

  tot_dyn <- sum(df_dyn[[col]])
  tot_s1  <- sum(df_raw[df_raw$n_sat == 1, col])
  tot_s2  <- sum(df_raw[df_raw$n_sat == 2, col])
  tot_s3  <- sum(df_raw[df_raw$n_sat == 3, col])

  # Build legend suffix (total or average depending on scale)
  use_avg <- max(df_raw[[col]]) < 200
  fmt_val <- function(v) if (use_avg) sprintf("%.2f avg", v/30) else
                                      sprintf("%s total", format(round(v), big.mark=","))

  lbl_dyn <- paste0("Dynamic optimal  (", fmt_val(tot_dyn), ")")
  lbl_s1  <- paste0("Fixed Sat=1  (",     fmt_val(tot_s1),  ")")
  lbl_s2  <- paste0("Fixed Sat=2  (",     fmt_val(tot_s2),  ")")
  lbl_s3  <- paste0("Fixed Sat=3  (",     fmt_val(tot_s3),  ")")

  # remap factor labels to include totals
  df_wide2 <- df_wide %>%
    mutate(n_sat2 = recode(as.character(n_sat),
                           "Fixed Sat=1" = lbl_s1,
                           "Fixed Sat=2" = lbl_s2,
                           "Fixed Sat=3" = lbl_s3))

  lt2 <- setNames(c("dashed","dotted","longdash"),              c(lbl_s1, lbl_s2, lbl_s3))
  cl2 <- setNames(c("#2288FF", "#FF6600", "#22CC44"),           c(lbl_s1, lbl_s2, lbl_s3))

  df_dyn2 <- df_dyn %>%
    mutate(dyn_lbl = lbl_dyn)

  p <- ggplot() +
    # fixed-sat lines
    geom_line(data = df_wide2,
              aes(x = day, y = .data[[col]], linetype = n_sat2, colour = n_sat2),
              linewidth = 0.7, alpha = 0.8) +
    scale_linetype_manual(values = lt2, breaks = c(lbl_s1, lbl_s2, lbl_s3)) +
    scale_colour_manual(
      values = c(cl2,
                 "Optimal = Sat 1" = "#2288FF",
                 "Optimal = Sat 2" = "#FF6600",
                 "Optimal = Sat 3" = "#22CC44"),
      breaks = c(lbl_s1, lbl_s2, lbl_s3)
    ) +

    # dynamic line (new aes mapping — separate scale via ggnewscale would be
    # cleaner, but we annotate manually to avoid the dependency)
    geom_line(data = df_dyn2,
              aes(x = day, y = .data[[col]]),
              colour = "black", linewidth = 1.1, linetype = "solid") +

    # optimal-satellite points on dynamic line
    geom_point(data = df_dyn,
               aes(x = day, y = .data[[col]], shape = opt_label, colour = opt_label),
               size = 2.8, stroke = 0.4) +
    scale_shape_manual(values = pt_shapes,
                       breaks = c("Optimal = Sat 1","Optimal = Sat 2","Optimal = Sat 3")) +

    # dynamic line legend entry (invisible geom)
    geom_line(data = data.frame(day=1, v=NA_real_, lbl=lbl_dyn),
              aes(x=day, y=v, linetype=lbl, colour=lbl),
              linewidth=1.1) +

    scale_x_continuous(breaks = 1:30, expand = expansion(add = 0.5)) +
    scale_y_continuous(labels = y_fmt) +
    labs(x = "Planning Day", y = y_label, title = title,
         linetype = NULL, colour = NULL, shape = NULL) +
    guides(
      linetype = guide_legend(order = 1, override.aes = list(linewidth = 0.8)),
      colour   = guide_legend(order = 1),
      shape    = guide_legend(order = 2)
    ) +
    base_theme

  ggsave(file.path(out_dir, out_file), plot = p,
         device = "pdf", width = 10, height = 5, units = "in")
  cat("Saved:", file.path(out_dir, out_file), "\n")
}

# ── generate four PDFs ────────────────────────────────────────────────────────
make_plot("cost_avg",
          "Total Cost ($)",
          "30-Day Cost Trend: Dynamic vs. Fixed Satellite Configurations",
          "case_cost_trend_r.pdf",
          y_fmt = comma)

make_plot("nv_avg",
          "Number of Vehicles (NV)",
          "30-Day Vehicle Count Trend: Dynamic vs. Fixed Satellite Configurations",
          "case_nv_trend_r.pdf",
          y_fmt = number_format(accuracy = 0.1))

make_plot("co2_avg",
          expression(CO[2]~Emissions~(kg)),
          "30-Day Carbon Emission Trend: Dynamic vs. Fixed Satellite Configurations",
          "case_co2_trend_r.pdf",
          y_fmt = comma)

make_plot("rt_avg",
          "Runtime (s)",
          "30-Day Runtime Trend: Dynamic vs. Fixed Satellite Configurations",
          "case_rt_trend_r.pdf",
          y_fmt = number_format(accuracy = 0.1))
