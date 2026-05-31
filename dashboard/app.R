# Financial Data Integrity Auditor - R Shiny dashboard.
#
# Modern card-based layout with bslib (Bootstrap 5), animated value boxes,
# loading spinners, and a playable time-axis on the Plotly discrepancy
# chart. Reads pre-computed CSVs from ../data/processed and ../data/splits
# - never calls a provider API directly.

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(ggplot2)
  library(DT)
  library(dplyr)
  library(readr)
  library(plotly)
  library(shinycssloaders)
  library(shinyjs)
  library(htmltools)
  library(scales)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

.app_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) NULL) %||% "."
PROJECT_ROOT <- normalizePath(file.path(.app_dir, ".."), mustWork = FALSE)
if (!dir.exists(file.path(PROJECT_ROOT, "data"))) {
  PROJECT_ROOT <- normalizePath("..", mustWork = FALSE)
}
if (!dir.exists(file.path(PROJECT_ROOT, "data"))) {
  PROJECT_ROOT <- normalizePath(".", mustWork = FALSE)
}
PROCESSED_DIR <- file.path(PROJECT_ROOT, "data", "processed")
SPLITS_DIR    <- file.path(PROJECT_ROOT, "data", "splits")

# Dynamic ticker universe: any ticker that has a *_comparison.csv on disk.
discover_tickers <- function() {
  files <- list.files(PROCESSED_DIR, pattern = "_comparison\\.csv$", full.names = FALSE)
  if (length(files) == 0) return(c("TSLA", "AAPL", "AMZN", "GME", "NVDA"))
  sort(sub("_comparison\\.csv$", "", files))
}
TICKERS <- discover_tickers()

# ---------------------------------------------------------------- helpers ----

load_comparison <- function(ticker) {
  path <- file.path(PROCESSED_DIR, paste0(ticker, "_comparison.csv"))
  if (!file.exists(path)) return(NULL)
  df <- readr::read_csv(path, show_col_types = FALSE)
  names(df)[1] <- "date"
  df$date <- as.Date(df$date)
  df
}

load_audit <- function(ticker) {
  path <- file.path(PROCESSED_DIR, paste0(ticker, "_audit.csv"))
  if (!file.exists(path)) return(NULL)
  df <- readr::read_csv(path, show_col_types = FALSE)
  if (nrow(df) > 0) df$date <- as.Date(df$date)
  df
}

load_splits <- function(ticker) {
  path <- file.path(SPLITS_DIR, paste0(ticker, "_splits.csv"))
  if (!file.exists(path)) return(NULL)
  df <- readr::read_csv(path, show_col_types = FALSE)
  names(df)[1] <- "date"
  df$date <- as.Date(df$date)
  df
}

# ----------------------------------------------------------------- theme ----

NAVY   <- "#0B1E3A"
AMBER  <- "#F59E0B"
CRIM   <- "#DC2626"
GREEN  <- "#10B981"
INK    <- "#111827"
PAPER  <- "#F8FAFC"
MUTED  <- "#6B7280"

app_theme <- bs_theme(
  version    = 5,
  bg         = PAPER,
  fg         = INK,
  primary    = NAVY,
  secondary  = "#475569",
  success    = GREEN,
  warning    = AMBER,
  danger     = CRIM,
  base_font  = font_google("Inter"),
  heading_font = font_google("Inter", wght = "600"),
  "card-border-radius"   = "14px",
  "card-cap-bg"          = "#FFFFFF",
  "card-cap-padding-y"   = "0.9rem",
  "card-spacer-y"        = "1rem",
  "border-color"         = "#E2E8F0",
  "body-color"           = INK
)

custom_css <- HTML("
  body { background-color: #F1F5F9; }
  .navbar-brand-row {
    background: linear-gradient(90deg, #0B1E3A 0%, #1E3A8A 100%);
    color: white;
    padding: 18px 24px;
    border-radius: 0 0 18px 18px;
    box-shadow: 0 6px 18px rgba(11,30,58,0.15);
    margin-bottom: 18px;
  }
  .navbar-brand-row h1 {
    color: white; font-weight: 700; font-size: 1.55rem; margin: 0;
    letter-spacing: -0.01em;
  }
  .navbar-brand-row .subtitle {
    color: #93C5FD; font-weight: 400; font-size: 0.95rem; margin-top: 4px;
  }
  .card {
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(11,30,58,0.04);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    background: white;
  }
  .card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.06), 0 12px 24px rgba(11,30,58,0.10);
  }
  .card-header {
    font-weight: 600; font-size: 0.92rem; letter-spacing: 0.02em;
    text-transform: uppercase; color: #475569; background: white !important;
    border-bottom: 1px solid #E2E8F0;
  }
  .bslib-value-box .value-box-title {
    font-size: 0.78rem !important; letter-spacing: 0.06em;
    text-transform: uppercase; color: rgba(255,255,255,0.85) !important;
  }
  .bslib-value-box .value-box-value {
    font-size: 1.9rem !important; font-weight: 700 !important;
    letter-spacing: -0.02em;
  }
  .bslib-value-box .value-box-showcase { opacity: 0.85; }
  .nav-tabs .nav-link {
    color: #475569; font-weight: 500; border: none;
    border-bottom: 2px solid transparent; padding: 10px 18px;
    transition: color 0.15s, border-color 0.15s;
  }
  .nav-tabs .nav-link:hover { color: #0B1E3A; border-bottom-color: #CBD5E1; }
  .nav-tabs .nav-link.active {
    color: #0B1E3A !important; background: transparent !important;
    border: none !important; border-bottom: 2px solid #F59E0B !important;
    font-weight: 600;
  }
  .control-card .form-label,
  .control-card .control-label { font-weight: 600; color: #334155;
    font-size: 0.85rem; letter-spacing: 0.02em; text-transform: uppercase; }
  .footer-note {
    color: #64748B; font-size: 0.82rem; padding: 14px 4px 4px;
    border-top: 1px solid #E2E8F0; margin-top: 14px;
  }
  .legend-pill {
    display: inline-block; padding: 4px 10px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 500; margin-right: 6px;
  }
  .pill-red    { background: #FEE2E2; color: #B91C1C; }
  .pill-amber  { background: #FEF3C7; color: #B45309; }
  .pill-gray   { background: #F1F5F9; color: #475569; }
  .explainer-box {
    background: #F1F5F9;
    border-left: 3px solid #0B1E3A;
    border-radius: 8px;
    padding: 12px 16px 12px 18px;
    font-size: 0.86rem;
    color: #334155;
  }
  .explainer-title {
    font-weight: 600; color: #0B1E3A; margin-bottom: 6px;
    font-size: 0.92rem;
  }
  .explainer-list { margin: 4px 0 0 0; padding-left: 18px; }
  .explainer-list li { margin-bottom: 4px; line-height: 1.45; }
  .explainer-list b { color: #0B1E3A; }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .fade-in {
    animation: fadeUp 0.45s ease both;
  }
  .stagger-1 { animation-delay: 0.05s; }
  .stagger-2 { animation-delay: 0.12s; }
  .stagger-3 { animation-delay: 0.18s; }
  .stagger-4 { animation-delay: 0.24s; }
  .stagger-5 { animation-delay: 0.30s; }
")

# ------------------------------------------------------------------- UI ----

header_row <- div(
  class = "navbar-brand-row fade-in",
  h1("Financial Data Integrity Auditor"),
  div(class = "subtitle",
      "Cross-vendor audit: Yahoo Finance vs Stooq, reconciled against official corporate actions")
)

sidebar_card <- card(
  class = "control-card fade-in stagger-1",
  card_header(tagList(icon("sliders"), " Controls")),
  card_body(
    selectInput("ticker", "Ticker", choices = TICKERS, selected = "AMZN"),
    sliderInput(
      "dateRange",
      "Date range",
      min   = as.Date("2018-01-01"),
      max   = Sys.Date(),
      value = c(as.Date("2018-01-01"), Sys.Date()),
      timeFormat = "%Y-%m-%d"
    ),
    helpText(
      "Reads pre-computed CSVs from data/processed/. Run the Python ",
      "pipeline first (ingest -> compare -> audit -> report)."
    )
  )
)

kpi_row <- layout_columns(
  col_widths = c(3, 3, 3, 3),
  fill = FALSE,
  class = "fade-in stagger-2",
  value_box(
    title = "Total flags",
    value = textOutput("kpi_total"),
    showcase = bsicons::bs_icon("flag-fill"),
    theme = value_box_theme(bg = "#0B1E3A", fg = "white")
  ),
  value_box(
    title = "Unexplained",
    value = textOutput("kpi_unex"),
    showcase = bsicons::bs_icon("exclamation-octagon-fill"),
    theme = value_box_theme(bg = "#DC2626", fg = "white")
  ),
  value_box(
    title = "Avg severity (z)",
    value = textOutput("kpi_sev"),
    showcase = bsicons::bs_icon("graph-up"),
    theme = value_box_theme(bg = "#F59E0B", fg = "white")
  ),
  value_box(
    title = "Worst discrepancy",
    value = textOutput("kpi_worst"),
    showcase = bsicons::bs_icon("activity"),
    theme = value_box_theme(bg = "#10B981", fg = "white")
  )
)

main_panel <- card(
  class = "fade-in stagger-3",
  full_screen = TRUE,
  card_header(tagList(icon("chart-line"), " Audit views")),
  card_body(
    navset_tab(
      id = "panel",
      nav_panel(
        "1. Ticker explorer",
        br(),
        withSpinner(
          plotOutput("explorerPlot", height = "520px"),
          type = 8, color = NAVY, size = 0.8
        ),
        br(),
        uiOutput("explorerNote")
      ),
      nav_panel(
        "2. Discrepancy monitor",
        br(),
        div(
          class = "mb-2",
          span(class = "legend-pill pill-red",   "Unexplained"),
          span(class = "legend-pill pill-amber", "Corporate action window"),
          span(class = "legend-pill pill-gray",  "Normal"),
          checkboxInput("animate", "Animate by year (play button)", value = FALSE)
        ),
        withSpinner(
          plotlyOutput("discrepancyPlot", height = "520px"),
          type = 8, color = NAVY, size = 0.8
        ),
        br(),
        uiOutput("discrepancyNote")
      ),
      nav_panel(
        "3. Anomaly log",
        br(),
        div(
          class = "explainer-box",
          tagList(
            tags$div(class = "explainer-title",
                     tagList(icon("circle-info"), " How to read this table")),
            tags$ul(class = "explainer-list",
              tags$li(tags$b("flag_type = \"corporate action window\""),
                      " - the discrepancy fell within +/- 5 calendar days of an ",
                      "official split. The audit attempts to adjudicate which ",
                      "provider's adjustment math is closer to the truth."),
              tags$li(tags$b("flag_type = \"unexplained discrepancy\""),
                      " - no official corporate action sits within +/- 5 days, ",
                      "so the disagreement is most likely a vendor-side issue ",
                      "(dividend-timing skew, back-fill, rounding drift, sync lag) ",
                      "rather than a split-adjustment error."),
              tags$li(tags$b("provider_verdict = \"n/a\""),
                      " - we cannot adjudicate provider correctness without a ",
                      "known official ratio. Only ",
                      tags$em("corporate-action-window"),
                      " rows can produce a verdict."),
              tags$li(tags$b("official_ratio = blank"),
                      " - same reason: there is no matched official split, ",
                      "so there is no ratio to record. Cells are intentionally ",
                      "left empty rather than filled with a placeholder.")
            )
          )
        ),
        br(),
        withSpinner(
          DTOutput("anomalyTable"),
          type = 8, color = NAVY, size = 0.8
        ),
        br(),
        uiOutput("anomalyNote")
      )
    )
  )
)

footer <- div(
  class = "footer-note",
  HTML(paste0(
    "<b>Methodology:</b> per-day adjustment factor ",
    "<code>f[t] = adj_close[t] / adj_close[latest]</code>, flagged when ",
    "|z-score| > 3 or absolute disagreement > 1%. Audit reconciles each ",
    "flag against the official Yahoo splits feed within +/- 5 calendar ",
    "days. Sources: Yahoo Finance via <code>yfinance</code>, Stooq via ",
    "HTTP CSV endpoint."
  ))
)

ui <- page_fluid(
  theme = app_theme,
  useShinyjs(),
  tags$head(tags$style(custom_css)),
  header_row,
  layout_columns(
    col_widths = c(3, 9),
    sidebar_card,
    div(
      kpi_row,
      br(),
      main_panel
    )
  ),
  footer
)

# --------------------------------------------------------------- server ----

server <- function(input, output, session) {

  comparison_data <- reactive({
    df <- load_comparison(input$ticker)
    if (is.null(df)) return(NULL)
    df %>% dplyr::filter(date >= input$dateRange[1], date <= input$dateRange[2])
  })

  audit_data <- reactive({
    df <- load_audit(input$ticker)
    if (is.null(df) || nrow(df) == 0) return(df)
    df %>% dplyr::filter(date >= input$dateRange[1], date <= input$dateRange[2])
  })

  # ---------- KPI value boxes ----------

  output$kpi_total <- renderText({
    a <- audit_data()
    if (is.null(a)) "-" else as.character(nrow(a))
  })

  output$kpi_unex <- renderText({
    a <- audit_data()
    if (is.null(a) || nrow(a) == 0) return("0")
    as.character(sum(a$flag_type == "unexplained discrepancy"))
  })

  output$kpi_sev <- renderText({
    a <- audit_data()
    if (is.null(a) || nrow(a) == 0) return("-")
    formatC(mean(a$severity_score, na.rm = TRUE), format = "f", digits = 2)
  })

  output$kpi_worst <- renderText({
    a <- audit_data()
    if (is.null(a) || nrow(a) == 0) return("-")
    bps <- max(a$discrepancy, na.rm = TRUE) * 10000
    paste0(formatC(bps, format = "f", digits = 1), " bps")
  })

  # ---------- Panel 1: Ticker explorer ----------

  output$explorerPlot <- renderPlot({
    df <- comparison_data()
    validate(need(!is.null(df), paste0(
      "No comparison data found for ", input$ticker,
      ". Run `python -m src.compare` first."
    )))

    long <- dplyr::bind_rows(
      dplyr::transmute(df, date = date, factor = yahoo_adj_factor, Provider = "Yahoo Finance"),
      dplyr::transmute(df, date = date, factor = stooq_adj_factor, Provider = "Stooq")
    )

    audit <- audit_data()
    flagged_dates <- if (!is.null(audit) && nrow(audit) > 0) audit$date else as.Date(character(0))

    p <- ggplot(long, aes(x = date, y = factor, colour = Provider)) +
      geom_line(linewidth = 0.7, alpha = 0.9) +
      scale_colour_manual(values = c(
        "Yahoo Finance" = "#1E40AF",
        "Stooq"         = "#059669"
      )) +
      scale_y_continuous(labels = scales::number_format(accuracy = 0.01)) +
      labs(
        title    = paste0(input$ticker, " - cumulative adjustment factor by provider"),
        subtitle = "Vertical red dashes = days flagged by the audit",
        x = NULL, y = "adj_close[t] / adj_close[latest]"
      ) +
      theme_minimal(base_size = 13, base_family = "sans") +
      theme(
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = "#E2E8F0"),
        plot.title       = element_text(face = "bold", colour = NAVY, size = 16),
        plot.subtitle    = element_text(colour = MUTED, size = 11),
        legend.position  = "top",
        legend.title     = element_blank(),
        legend.text      = element_text(size = 12),
        axis.text        = element_text(colour = "#334155"),
        axis.title.y     = element_text(colour = MUTED, size = 10),
        plot.background  = element_rect(fill = "white", colour = NA),
        panel.background = element_rect(fill = "white", colour = NA)
      )

    if (length(flagged_dates) > 0) {
      p <- p + geom_vline(
        xintercept = as.numeric(flagged_dates),
        linetype = "dashed", colour = CRIM, alpha = 0.55, linewidth = 0.4
      )
    }
    p
  })

  output$explorerNote <- renderUI({
    df <- comparison_data()
    if (is.null(df)) return(NULL)
    audit <- audit_data()
    n_flags <- if (is.null(audit)) 0 else nrow(audit)
    HTML(paste0(
      "<span style='color:#475569'>",
      "<b>", nrow(df), "</b> trading days in view, <b>", n_flags,
      "</b> flagged. Healthy providers' adjustment-factor curves should ",
      "sit on top of each other; visible separation is a data-quality smell.",
      "</span>"
    ))
  })

  # ---------- Panel 2: Discrepancy monitor ----------

  output$discrepancyPlot <- renderPlotly({
    df <- comparison_data()
    validate(need(!is.null(df), paste0(
      "No comparison data found for ", input$ticker, "."
    )))

    audit <- audit_data()
    df$flag_type <- "normal"
    if (!is.null(audit) && nrow(audit) > 0) {
      df$flag_type[df$date %in% audit$date[audit$flag_type == "corporate action window"]] <-
        "corporate action window"
      df$flag_type[df$date %in% audit$date[audit$flag_type == "unexplained discrepancy"]] <-
        "unexplained"
    }
    df$year <- format(df$date, "%Y")

    pal <- c(
      "normal"                   = "#94A3B8",
      "corporate action window"  = AMBER,
      "unexplained"              = CRIM
    )
    df$flag_type <- factor(df$flag_type,
                           levels = c("normal", "corporate action window", "unexplained"))

    base_layout <- list(
      title = list(
        text = paste0("<b>", input$ticker, "</b> - per-day adjustment-factor disagreement"),
        font = list(family = "sans-serif", size = 16, color = NAVY),
        x = 0.02, xanchor = "left"
      ),
      xaxis = list(title = "", gridcolor = "#E2E8F0", showline = FALSE),
      yaxis = list(
        title = "|yahoo_adj_factor - stooq_adj_factor|",
        gridcolor = "#E2E8F0", showline = FALSE,
        titlefont = list(size = 11, color = MUTED)
      ),
      plot_bgcolor  = "white",
      paper_bgcolor = "white",
      font = list(family = "sans-serif", color = INK),
      shapes = list(list(
        type = "line",
        x0 = min(df$date), x1 = max(df$date),
        y0 = 0.01, y1 = 0.01,
        line = list(color = CRIM, dash = "dash", width = 1.2)
      )),
      legend = list(orientation = "h", y = 1.06, x = 0.4,
                    bgcolor = "rgba(0,0,0,0)")
    )

    if (isTRUE(input$animate)) {
      p <- plot_ly(
        df, x = ~date, y = ~discrepancy,
        type = "scatter", mode = "markers",
        color = ~flag_type, colors = pal,
        frame = ~year,
        marker = list(size = 7, line = list(width = 0)),
        hovertemplate = paste(
          "<b>%{x|%Y-%m-%d}</b><br>",
          "Discrepancy: %{y:.6f}",
          "<extra></extra>"
        )
      ) %>%
        animation_opts(frame = 700, transition = 350, redraw = TRUE) %>%
        animation_slider(currentvalue = list(prefix = "Year: ",
                                             font = list(color = NAVY))) %>%
        animation_button(label = "Play", visible = TRUE)
      do.call(layout, c(list(p), base_layout))
    } else {
      p <- plot_ly(
        df, x = ~date, y = ~discrepancy,
        type = "scatter", mode = "markers",
        color = ~flag_type, colors = pal,
        marker = list(size = 5, line = list(width = 0)),
        hovertemplate = paste(
          "<b>%{x|%Y-%m-%d}</b><br>",
          "Discrepancy: %{y:.6f}",
          "<extra></extra>"
        )
      )
      do.call(layout, c(list(p), base_layout))
    }
  })

  output$discrepancyNote <- renderUI({
    HTML(paste0(
      "<span style='color:#475569'>",
      "Dashed red line = 1% raw-threshold for flagging. Toggle <b>Animate ",
      "by year</b> above to watch the disagreement evolve chronologically.",
      "</span>"
    ))
  })

  # ---------- Panel 3: Anomaly log ----------

  output$anomalyTable <- renderDT({
    audit <- audit_data()
    validate(need(
      !is.null(audit) && nrow(audit) > 0,
      paste0("No flagged rows for ", input$ticker, " in this date range.")
    ))

    df <- audit %>%
      dplyr::select(
        date, flag_type, provider_verdict, official_ratio, severity_score
      ) %>%
      dplyr::arrange(dplyr::desc(severity_score))

    datatable(
      df,
      rownames  = FALSE,
      class     = "stripe hover compact",
      options   = list(
        pageLength = 15,
        order      = list(list(4, "desc")),
        dom        = "tip",
        columnDefs = list(list(className = "dt-center", targets = "_all"))
      ),
      selection = "none"
    ) %>%
      formatStyle(
        "flag_type",
        target = "row",
        backgroundColor = styleEqual(
          c("unexplained discrepancy", "corporate action window"),
          c("#FEF2F2", "#FFFBEB")
        )
      ) %>%
      formatStyle(
        "severity_score",
        fontWeight = "bold",
        color = styleInterval(c(5, 10, 20), c(MUTED, "#92400E", "#B91C1C", "#7F1D1D"))
      ) %>%
      formatRound("official_ratio", digits = 4) %>%
      formatRound("severity_score", digits = 2)
  })

  output$anomalyNote <- renderUI({
    audit <- audit_data()
    if (is.null(audit) || nrow(audit) == 0) return(NULL)
    n_unex <- sum(audit$flag_type == "unexplained discrepancy")
    HTML(paste0(
      "<span style='color:#475569'>",
      "<b>", nrow(audit), "</b> flagged rows for ", input$ticker, ". ",
      "<b>", n_unex, "</b> unexplained (no official corporate action ",
      "within +/- 5 calendar days) - the highest-priority incidents.",
      "</span>"
    ))
  })
}

shinyApp(ui, server)
